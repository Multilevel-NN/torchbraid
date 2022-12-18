"""
This file contains:
  - Generic multilevel solver.
    Based on PyAMG multilevel solver, https://github.com/pyamg/pyamg
    PyAMG is released under the MIT license.
  - MG/Opt implementations of the multilevel solver
"""


from __future__ import print_function

import numpy as np

import sys
import statistics as stats
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchbraid
import torchbraid.utils

from timeit import default_timer as timer

from mpi4py import MPI


__all__ = [ 'mgopt_solver','compute_levels','root_print' ]

####################################################################################
####################################################################################
# Functions to facilitate MG/Opt and PyTorch 

##
# Basic Linear algebra functions
def tensor_list_dot(v, w, comm):
  ''' Compute dot product of two vectors, v and w, where each vector is a list of tensors '''
  my_sum = sum([ torch.dot(vv.flatten(), ww.flatten()) for (vv,ww) in zip(v, w) ])
  # For parallel, we just fill my_sum with the global inner-product value (without updating the autograd tape)
  # We assume that this dot-product operation is only ever used for "linear" operations, like the <x_h, v_h> 
  # term inside of MG/Opt, so that this little trick will work.
  if comm.Get_size() > 1:
    global_sum = comm.allreduce(my_sum.item(), op=MPI.SUM)
    with torch.no_grad(): 
      my_sum.data.fill_(global_sum)

  return my_sum

def tensor_list_AXPY(alpha, v, beta, w, inplace=False):
  '''
  Compute AXPY two vectors, v and w, where each vector is a list of tensors
  if inplace is True, then w = alpha*v + beta*w
  else, return a new vector equal to alpha*v + beta*w 
  '''

  if inplace:
    for (vv, ww) in zip(v, w):
      ww[:] = alpha*vv + beta*ww
  else:
    return [ alpha*vv + beta*ww for (vv,ww) in zip(v, w) ]

def tensor_list_deep_copy(w):
  ''' return a deep copy of the tensor list w '''
  return [ torch.clone(ww) for ww in w ]


##
# PyTorch train and test network functions
def train_epoch(rank, model, train_loader, optimizer, epoch, criterion, criterion_kwargs, compose, log_interval, device, mgopt_printlevel):
  '''
  Carry out one complete training epoch
  If "optimizer" is a tuple, use it to create a new optimizer object each batch (i.e., reset the optimizer state each batch)
  Else, just use optimizer to take steps (assume it is a PyTorch optimizer object)
  '''
  model.train()
  
  total_time = 0.0
  for batch_idx, (data, target) in enumerate(train_loader):
    data = data.to(device)
    target = target.to(device)

    start_time = timer()
    #optimizer = optim.Adam(model.parameters(), **{ 'lr':0.001, 'betas':(0.9, 0.999), 'eps':1e-08 })
    # Allow for optimizer to be reset each batch, or not
    if type(optimizer) == tuple:
      (optim, optim_kwargs) = mgopt_solver().process_optimizer(optimizer, model)
    else:
      optim = optimizer
    ##
    optim.zero_grad()
    output = model(data)

    loss = compose(criterion, output, target, **criterion_kwargs)
    loss.backward()
    stop_time = timer()
    optim.step()

    total_time += stop_time-start_time

    
    root_print(rank, mgopt_printlevel, 2, "Batch:  " + str(batch_idx) + "    Loss = " + str(loss.item()) )
    if batch_idx % log_interval == 0:
      root_print(rank, mgopt_printlevel, 2, "\n------------------------------------------------------------------------------")
      root_print(rank, mgopt_printlevel, 1, '  Train Epoch: {} [{}/{} ({:.0f}%)]     \tLoss: {:.9f}\tTime Per Batch {:.6f}'.format(
          epoch, batch_idx * len(data), len(train_loader.dataset),
          100. * batch_idx / len(train_loader), loss.item(), total_time/(batch_idx+1.0)))
      root_print(rank, mgopt_printlevel, 2, "------------------------------------------------------------------------------\n")
  ##

  root_print(rank, mgopt_printlevel, 2, "\n------------------------------------------------------------------------------")
  root_print(rank, mgopt_printlevel, 1, '  Train Epoch: {} [{}/{} ({:.0f}%)]     \tLoss: {:.9f}\tTime Per Batch {:.6f}'.format(
    epoch, (batch_idx+1) * len(data), len(train_loader.dataset),
    100. * (batch_idx+1) / len(train_loader), loss.item(), total_time/(batch_idx+1.0)))
  root_print(rank, mgopt_printlevel, 2, "------------------------------------------------------------------------------\n")


def test(rank, model, test_loader, criterion, criterion_kwargs, compose, device, mgopt_printlevel, indent=''):
  ''' Compute loss and accuracy '''
  comm = model.parallel_nn.fwd_app.mpi_comm
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      output = model(data)
      test_loss += compose(criterion, output, target, **criterion_kwargs).item()
       
      output = comm.bcast(output,root=0)
      pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
      correct += pred.eq(target.view_as(pred)).sum().item()

  test_loss /= len(test_loader.dataset)

  root_print(rank, mgopt_printlevel, 1, indent + 'Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:2.2f}%)\n'.format(
      test_loss, correct, len(test_loader.dataset),
      100. * correct / len(test_loader.dataset)))


def compute_fwd_bwd_pass(lvl, optimizer, model, data, target, criterion, criterion_kwargs, compose, v_h):
  '''
  Compute a backward and forward pass for the model.
  if lvl is 0, no MGOPT term is used
  if lvl > 0, incorporate MGOpt term

  returns the loss as a scalar (i.e., with no tape attached)
  '''
  model.train()

  optimizer.zero_grad()
  output = model(data)

  if lvl == 0:
    loss = compose(criterion, output, target, **criterion_kwargs)
  else: # add the MG/Opt Term
    x_h = get_params(model, deep_copy=False, grad=False)
    mgopt_term = tensor_list_dot(v_h, x_h, model.parallel_nn.fwd_app.mpi_comm)
    loss = compose(criterion, output, target, mgopt_term=mgopt_term, **criterion_kwargs)
  ##
  loss.backward()

  # Loss is only available on rank 0
  comm = model.parallel_nn.fwd_app.mpi_comm
  loss_scalar = comm.bcast(loss.item(), root=0)

  return loss_scalar


def compute_fwd_pass(lvl, model, data, target, criterion, criterion_kwargs, compose, v_h):
  '''
  Compute a forward pass only to obtain a loss for the model.
  if lvl is 0, no MGOPT term is used
  if lvl > 0, incorporate MGOpt term

  returns the loss as a scalar (i.e., with no tape attached)
  '''
  model.eval()
  output = model(data)

  if lvl == 0:
    loss = compose(criterion, output, target, **criterion_kwargs)
  else: # add the MG/Opt Term
    x_h = get_params(model, deep_copy=False, grad=False)
    mgopt_term = tensor_list_dot(v_h, x_h, model.parallel_nn.fwd_app.mpi_comm)
    loss = compose(criterion, output, target, mgopt_term=mgopt_term, **criterion_kwargs)
  ##

  # Loss is only available on rank 0
  comm = model.parallel_nn.fwd_app.mpi_comm
  loss_scalar = comm.bcast(loss.item(), root=0)

  return loss_scalar



##
# PyTorch get and write params functions
def write_params_inplace(model, new_params, grad=False):
  '''
  Write the parameters of model in-place, overwriting with new_params
  '''
  
  with torch.no_grad():
    old_params = list(model.parameters())
    
    assert(len(old_params) == len(new_params)) 
    
    for (op, np) in zip(old_params, new_params):
      if grad: op.grad[:] = np[:]
      else:    op[:] = np[:]

def get_params(model, deep_copy=False, grad=False):
  '''
  Get the network parameters
  '''
  if deep_copy:
    if grad: pp = [torch.clone(params.grad) for params in model.parameters() ]
    else:    pp = [torch.clone(params)      for params in model.parameters() ]
  else:
    if grad: pp = [params.grad for params in model.parameters() ]
    else:    pp = [params      for params in model.parameters() ]
  ##

  return pp


def get_adam_momentum(model, optimizer):
  '''
  Get the momentum term out of an Adam optimizer
  '''
  momentum = []
  for pp in model.parameters():
    if pp.grad is not None:
      # Grab the momentum for this parameter
      momentum.append( torch.clone( optimizer.state[pp]['exp_avg'] ))
      #momentum.append( torch.clone( optimizer.state[pp]['exp_avg'] / (np.sqrt(optimizer.state[pp]['exp_avg_sq']) + 1e-8)   ))
  ##
  return momentum
 

####################################################################################
####################################################################################



####################################################################################
####################################################################################
# TorchBraid Interp / restrict functions


def tb_parallel_get_injection_interp_params(model_fine, model_coarse, cf=2, deep_copy=True, grad=False):
  ''' 
  Interpolate the model parameters according to coarsening-factor in time cf.

  This is carried out in parallel (requires PYX Cython layer), hence call to helper function
  
  Return a list of the interpolated model parameters on this processor
  If grad is True, return the network gradient instead
  '''
 
  if deep_copy == False:
    raise ValueError('tb_parallel_get_injection_interp_params  does not support deep_copy==False')

  fwd_app = model_fine.parallel_nn.fwd_app
  interp_params = fwd_app.parallel_injection_interp_params(model_fine, model_coarse, cf=cf, grad=grad)
  return interp_params

def tb_get_injection_interp_params(model_fine, model_coarse, cf=2, deep_copy=False, grad=False):
  
  ''' 
  Interpolate the model_coarse parameters according to coarsening-factor in time cf.
  Return a list of the interpolated model parameters.

  If deep_copy is True, return a deep copy.
  If grad is True, return the network gradient instead
  '''
  # See
  # https://stackoverflow.com/questions/383565/how-to-iterate-over-a-list-repeating-each-element-in-python
  def duplicate(iterable,n):
    """A generator that repeats each entry n times"""
    for item in iterable:
      first = True
      for _ in range(n):
        yield item,first
        first = False

  interp_params = []

  # loop over all the children, interpolating the layer-parallel weights
  with torch.no_grad():
    for child in model_coarse.children():
      # handle layer parallel modules differently 
      if isinstance(child, torchbraid.LayerParallel):
        # loop over each layer-parallel layer -- this is where the "interpolation" occurs
        for (lp_child, lp_f) in duplicate(child.layer_models, cf):
          for param in lp_child.parameters():
            if deep_copy:
              if grad: interp_params.append(torch.clone(param.grad))
              else:    interp_params.append(torch.clone(param))
            else: # no deep copy
              if grad: interp_params.append(param.grad)
              else:    interp_params.append(param)
               
      else:
        # Do simple injection for the opening and closing layers
        for param in child.parameters():
          if deep_copy:
            if grad: interp_params.append(torch.clone(param.grad))
            else:    interp_params.append(torch.clone(param))
          else: # no deep copy
            if grad: interp_params.append(param.grad)
            else:    interp_params.append(param)
    ##

  return interp_params

def tb_get_linear_interp_params(model_fine, model_coarse, cf=2, deep_copy=True, grad=False):
  
  ''' 
  Interpolate the model_coarse parameters according to coarsening-factor in time cf.
  Return a list of the interpolated model parameters.

  If deep_copy is True, return a deep copy.
  If grad is True, return the network gradient instead
  '''

  interp_params = []
  
  def create_parameter_copy(layer, grad):
    ''' return a deep copy of this layer's parameters in list form '''
    param_copy = []
    for param in layer.parameters():
      if grad: param_copy.append(torch.clone(param.grad))
      else:    param_copy.append(torch.clone(param))
    #
    return param_copy

  def create_parameter_linear_combo(l_i, l_j, w_i, w_j, grad):
    ''' return a deep copy of the parameters for:  w_i*l_i + w_j*l_j '''
    param_copy = []
    for (p_i, p_j) in zip(l_i.parameters(), l_j.parameters()):
      if grad: param_copy.append( w_i*p_i.grad + w_j*p_j.grad )
      else:    param_copy.append( w_i*p_i + w_j*p_j )
    #
    return param_copy


  if deep_copy == False:
    print("deep_copy False not supported for linear interp.  copying unavoidable")
    assert(False)
  
  # loop over all the children, interpolating the layer-parallel weights
  with torch.no_grad():
    for child in model_coarse.children():
        
      # handle layer parallel modules differently 
      if isinstance(child, torchbraid.LayerParallel):
        nlayers = len(child.layer_models)
        
        # copy the first layer
        layer0 = child.layer_models[0]
        interp_params = interp_params + create_parameter_copy(layer0, grad)

        # interp intermediate layers
        for i in range(1, nlayers):
          left = child.layer_models[i-1]
          right = child.layer_models[i]
          for j in range(1, cf):
            weight_l = (cf - j)*(1.0/cf)
            weight_r = j*(1.0/cf)
            interp_params = interp_params + create_parameter_linear_combo(left, right, weight_l, weight_r, grad)
          
          # finally, insert a full copy of right point
          interp_params = interp_params + create_parameter_copy(right, grad)

               
        # copy the dangling F-points at end of time-line, use piece-wise constant 
        layer = child.layer_models[-1]
        for i in range(cf-1):
          interp_params = interp_params + create_parameter_copy(layer, grad)
        
      else:
        # Do simple injection for the opening and closing layers
        for param in child.parameters():
          if grad: interp_params.append(torch.clone(param.grad))
          else:    interp_params.append(torch.clone(param))
    ##

  return interp_params



def tb_parallel_get_injection_restrict_params(model_fine, model_coarse, cf=2, deep_copy=True, grad=False):
  ''' 
  Restrict the model_fine parameters according to coarsening-factor in time cf.

  This is carried out in parallel (requires PYX Cython layer), hence call to helper function
  
  Return a list of the restricted model parameters.
  If grad is True, return the network gradient instead
  '''
 
  if deep_copy == False:
    raise ValueError('tb_parallel_get_injection_restrict_params  does not support deep_copy==False')

  fwd_app = model_fine.parallel_nn.fwd_app
  restrict_params = fwd_app.parallel_injection_restrict_params(model_fine, model_coarse, cf=cf, grad=grad)
  return restrict_params


def tb_get_injection_restrict_params(model_fine, model_coarse, cf=2, deep_copy=False, grad=False):
  ''' 
  Restrict the model_fine parameters according to coarsening-factor in time cf.
  Return a list of the restricted model parameters.

  If deep_copy is True, return a deep copy.
  If grad is True, return the network gradient instead
  '''
  
  restrict_params = []

  # loop over all the children, restricting 
  for child in model_fine.children():
    
    # handle layer parallel modules differently 
    if isinstance(child, torchbraid.LayerParallel):
      
      # loop over each layer-parallel layer -- this is where the "restriction" occurs
      for lp_child in child.layer_models[0:-1:cf]: 
        with torch.no_grad():
          for param in lp_child.parameters(): 
            if deep_copy:
              if grad: restrict_params.append(torch.clone(param.grad))
              else:    restrict_params.append(torch.clone(param))
            else: # no deep copy
              if grad: restrict_params.append(param.grad)
              else:    restrict_params.append(param)
             
    else:
      # Do simple injection for the opening and closing layers
      with torch.no_grad():
        for param in child.parameters(): 
          if deep_copy:
            if grad: restrict_params.append(torch.clone(param.grad))
            else:    restrict_params.append(torch.clone(param))
          else: # no deep copy
            if grad: restrict_params.append(param.grad)
            else:    restrict_params.append(param)
    ##

  return restrict_params 

def tb_get_linear_restrict_params(model_fine, model_coarse, cf=2, deep_copy=True, grad=False):
  ''' 
  Restrict the model_fine parameters according to coarsening-factor in time cf.
  Return a list of the restricted model parameters.

  If deep_copy is True, return a deep copy.
  If grad is True, return the network gradient instead
  '''
  
  restrict_params = []
  
  stencil_avg = [0.25,  0.5,  0.25] 
  stencil_inj = [None,  1.0,  None] 

  def combine(left, center, right, sten, grad):
    ''' Helper function to combine three layers, weighted by the stencil in sten'''
    new_params = []
    with torch.no_grad():
      for param in center.parameters():
        if grad: new_params.append(sten[1]*torch.clone(param.grad))
        else: new_params.append(sten[1]*torch.clone(param))
      if left != None:
        for idx, param in enumerate(left.parameters()):
          if grad: new_params[idx] = new_params[idx] + sten[0]*param.grad
          else: new_params[idx] = new_params[idx] + sten[0]*param
      if right != None:
        for idx, param in enumerate(right.parameters()):
          if grad: new_params[idx] = new_params[idx] + sten[2]*param.grad
          else: new_params[idx] = new_params[idx] + sten[2]*param
    ##
    return new_params    
        

  if deep_copy == False:
    print("deep_copy False not supported for linear restrict.  copying unavoidable")
    assert(False)
  
  # loop over all the children, restricting 
  for child in model_fine.children():
    
    # handle layer parallel modules differently 
    if isinstance(child, torchbraid.LayerParallel):
      nlayers = len(child.layer_models)
      num_Cpts = int((nlayers - 1)/cf + 1)
      last_Cpt = (num_Cpts - 1)*cf  # computes index
      
      if(nlayers < 2):
        print("linear restrict requires at least 2 layers") 
        assert(False)
      
      # restrict first layer
      with torch.no_grad():
        center = child.layer_models[0]
        restrict_params = restrict_params + combine(None, center, None, stencil_inj, grad)

      # restrict middle layers
      for idx in range(cf, nlayers-cf, cf):
        left = child.layer_models[idx-1]
        center = child.layer_models[idx]
        right = child.layer_models[idx+1]
        restrict_params = restrict_params + combine(left, center, right, stencil_avg, grad)

      # restrict last C-point
      if last_Cpt > 1:
        # Special case if last C-point is last point on the grid
        if last_Cpt == (nlayers - 1):
          assert(False)
          # This shouldn't be reached
          with torch.no_grad():
            center = child.layer_models[last_Cpt]
            restrict_params = restrict_params + combine(None, center, None, stencil_inj, grad)
         
        # Last C-point is not last point on the grid
        else:
          with torch.no_grad():
            center = child.layer_models[last_Cpt]
            left = child.layer_models[last_Cpt-1]
            right = child.layer_models[last_Cpt+1]
            restrict_params = restrict_params + combine(left, center, right, stencil_avg, grad)


    else:
      # Do simple injection for the opening and closing layers
      with torch.no_grad():
        for param in child.parameters(): 
          if grad: restrict_params.append(torch.clone(param.grad))
          else:    restrict_params.append(torch.clone(param))

    
  return restrict_params 


def tb_injection_restrict_network_state(model_fine, model_coarse, cf=2):
  ''' 
  Restrict the model state according to coarsening-factor in time cf.
  The restricted model state is placed inside of model_coarse
  '''
  
  ##
  # Inject network state to model_coarse.  
  # Note:  We need access to cython cdef to do this, so we put the restrict/interp inside torchbraid_app.pyx
  model_coarse.parallel_nn.fwd_app.inject_network_state(  model_fine.parallel_nn.fwd_app, cf )  
  model_coarse.parallel_nn.bwd_app.inject_network_state(  model_fine.parallel_nn.bwd_app, cf )  


def tb_injection_interp_network_state(model_fine, model_coarse, cf=2):
  ''' 
  Interp the model state according to coarsening-factor in time cf.
  The interpolated model state is placed inside of model_fine 
  '''
  
  ##
  # interp network state to model_fine.  
  # Note:  We need access to cython cdef to do this, so we put the restrict/interp inside torchbraid_app.pyx
  model_fine.parallel_nn.fwd_app.interp_network_state(  model_coarse.parallel_nn.fwd_app, cf )  
  model_fine.parallel_nn.bwd_app.interp_network_state(  model_coarse.parallel_nn.bwd_app, cf )  

def tb_injection_restrict_adam_state(model_fine, model_coarse, opt_fine, opt_coarse, cf=2):
  '''
  Restrict the Adam optimizer state from fine-grid optimizer (opt_fine) to
  coarse-grid optimizer (opt_coarse)
  '''
  
  ##
  # The parameters themselves serve as the "dictionary" keys into the Adam
  # optimizer state.  Thus, we need structured lists of these parameters "indices"
  #
  # Opening layer lists for model_fine and model_coarse
  ol_fine = []
  ol_coarse = []
  # Closing layer
  cl_fine = []
  cl_coarse = []
  # Layer parallel
  lp_fine = []
  lp_coarse = []
  
  ##
  # Loop over all coarse layers
  with torch.no_grad():
    for child in model_coarse.children():
      
      # Store as opening, closing, or LP layer
      name = str(type(child))
      name = name[ name.rfind('.')+1 : name.rfind('\'') ]
      if name == 'LayerParallel':

        for layer in child.layer_models: 
          # Grab this layer's parameters
          this_layer = [] 
          for param in layer.parameters():
            this_layer.append(param)
          ##
          lp_coarse.append(this_layer)

      else:

        # Grab this layer's parameters
        this_layer = [] 
        for param in child.parameters():
          this_layer.append(param)

        if name == 'CloseLayer':
          cl_coarse = this_layer
        elif name == 'OpenLayer':
          ol_coarse = this_layer
        else:
          output_exception('Layer type needs to be OpenLayer, CloseLayer, or LayerParallel')

  ##
  # Loop over all fine layers
  with torch.no_grad():
    for child in model_fine.children():
      
      # Store as opening, closing, or LP layer
      name = str(type(child))
      name = name[ name.rfind('.')+1 : name.rfind('\'') ]
      if name == 'LayerParallel':

        for layer in child.layer_models: 
          # Grab this layer's parameters
          this_layer = [] 
          for param in layer.parameters():
            this_layer.append(param)
          ##
          lp_fine.append(this_layer)

      else:

        # Grab this layer's parameters
        this_layer = [] 
        for param in child.parameters():
          this_layer.append(param)

        if name == 'CloseLayer':
          cl_fine = this_layer
        elif name == 'OpenLayer':
          ol_fine = this_layer
        else:
          output_exception('Layer type needs to be OpenLayer, CloseLayer, or LayerParallel')

  ##
  # Sanity check on network sizes
  if( len(cl_fine) != len(cl_coarse)):
      raise ValueError('Fine and coarse closing layer not compatible in size')
  if( len(ol_fine) != len(ol_coarse)):
      raise ValueError('Fine and coarse opening layer not compatible in size')
  if( len(lp_fine)/cf != len(lp_coarse)):
      raise ValueError('Fine and coarse ODE layer parallel not compatible in size')
  
  ##       
  # Copy opening and closing layers
  for pfine, pcoarse in zip(cl_fine, cl_coarse):
    (opt_coarse.state[pcoarse])['exp_avg'][:] = (opt_fine.state[pfine])['exp_avg'][:]
    (opt_coarse.state[pcoarse])['exp_avg_sq'][:] = (opt_fine.state[pfine])['exp_avg_sq'][:]
    (opt_coarse.state[pcoarse])['step'] = (opt_fine.state[pfine])['step']
  
  for pfine, pcoarse in zip(ol_fine, ol_coarse):
    (opt_coarse.state[pcoarse])['exp_avg'][:] = (opt_fine.state[pfine])['exp_avg'][:]
    (opt_coarse.state[pcoarse])['exp_avg_sq'][:] = (opt_fine.state[pfine])['exp_avg_sq'][:]
    (opt_coarse.state[pcoarse])['step'] = (opt_fine.state[pfine])['step']
  
  ##
  # Copy over the coarsened ODE layer parallel
  for layer_fine, layer_coarse in zip(lp_fine[::cf], lp_coarse):
    for pfine, pcoarse in zip(layer_fine, layer_coarse):
      (opt_coarse.state[pcoarse])['exp_avg'][:] = (opt_fine.state[pfine])['exp_avg'][:]
      (opt_coarse.state[pcoarse])['exp_avg_sq'][:] = (opt_fine.state[pfine])['exp_avg_sq'][:]
      (opt_coarse.state[pcoarse])['step'] = (opt_fine.state[pfine])['step']






##
# Basic TorchBraid cross Entropy loss function extended to take MG/Opt Term
def tb_mgopt_cross_ent(output, target, mgopt_term=None, model=None):
  '''
  Define cross entropy loss with optional new MG/Opt term
  '''
  if False:
    # Manually define basic cross-entropy (probably not useful)
    log_prob = -1.0 * F.log_softmax(output, 1)
    loss = log_prob.gather(1, target.unsqueeze(1))
    loss = loss.mean()
  else:
    # Use PyTorch's loss, which should be identical
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, target)
  
  # Compute MGOPT term (be careful to use only PyTorch functions)
  if mgopt_term != None: 
    return loss - mgopt_term
  else:
    return loss

##
# Basic TorchBraid regression loss function extended to take MG/Opt Term
def tb_mgopt_regression(output, target, network_parameters=None, v=None, model=None):
  '''
  Define cross entropy loss with optional new MG/Opt term
  '''
  # Use PyTorch's loss, which should be identical
  criterion = nn.MSELoss()
  loss = criterion(output, target)
  
  # Compute MGOPT term (be careful to use only PyTorch functions)
  if (network_parameters is not None) and (v is not None): 
    mgopt_term = tensor_list_dot(v, network_parameters)
    return loss - mgopt_term
  else:
    return loss


##
# Basic TorchBraid cross Entropy loss function extended in two ways (i) to take MG/Opt Term, and (ii) to add continuity regularization terms
def tb_mgopt_cross_ent_plus_continuity(output, target, mgopt_term=None, model=None, continuity_constant=0.0001):
  '''
  Define cross entropy loss with optional new MG/Opt term
  '''
  if False:
    # Manually define basic cross-entropy (probably not useful)
    log_prob = -1.0 * F.log_softmax(output, 1)
    loss = log_prob.gather(1, target.unsqueeze(1))
    loss = loss.mean()
  else:
    # Use PyTorch's loss, which should be identical
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, target)

  
  # Add regularization terms for continuity and on size of classification layer
  dt = model.parallel_nn.fwd_app.dt
  # First deal with continuity term
  scaling = continuity_constant/(4.0*dt)
  for child in model.children():
    
    # The three NN blocks are stored as opening, closing, or LP layer
    name = str(type(child))
    name = name[ name.rfind('.')+1 : name.rfind('\'') ]
    if name == 'LayerParallel':

      for (left,right) in zip( child.layer_models[:-1], child.layer_models[1:]): 
        for (p_left, p_right) in zip(left.parameters(), right.parameters()):
          diff = (p_left - p_right).flatten()
          loss = loss + scaling*torch.dot(diff, diff)
  

  ## Second comes the second derivative term
  #scaling = continuity_constant/(4.0*dt*dt)
  #for child in model.children():
  #  
  #  # The three NN blocks are stored as opening, closing, or LP layer
  #  name = str(type(child))
  #  name = name[ name.rfind('.')+1 : name.rfind('\'') ]
  #  if name == 'LayerParallel':
  #
  #    for (left, center, right) in zip( child.layer_models[:-2], child.layer_models[1:-1], child.layer_models[2:]): 
  #      for (p_left, p_center, p_right) in zip(left.parameters(), center.parameters(), right.parameters()):
  #        diff = (p_left - 2.0*p_center + p_right).flatten()
  #        loss = loss + scaling*torch.dot(diff, diff)
  #
  ###
  ## Third deal with classification layer
  #scaling = continuity_constant/4.0
  #for child in model.children():
  #  
  #  # The three NN blocks are stored as opening, closing, or LP layer
  #  name = str(type(child))
  #  name = name[ name.rfind('.')+1 : name.rfind('\'') ]
  #  if name == 'CloseLayer':
  #    for param in child.parameters():
  #      loss = loss + scaling*torch.dot(param.flatten(), param.flatten())


  # Compute MGOPT term (be careful to use only PyTorch functions)
  if mgopt_term != None: 
    return loss - mgopt_term
  else:
    return loss



def tb_adam_no_ls(lvl, e_h, x_h, v_h, model, data, target, optimizer, criterion, criterion_kwargs, compose, old_loss, e_dot_gradf, mgopt_printlevel, ls_params):
  '''
  Do no line search (save computation) and simply incorporate the coarse-grid
  correction as a fine-level optimizer (Adam, SGD, ...) update.  This requires
  simply overwriting the gradient with the coarse-grid correction direction,
  -e_h.  Note, the standard coarse-grid update adds e_h to the parameters, so
  e_h already represents a "negative" gradient direction. 
  '''
  
  ##
  # Not clear how to use Adam as our mediator for the coarse-grid correction.
  # Thus, we have some commented out possibilities regarding changing the beta
  # values and preserving the Adam momentum state (i.e., do not allow the
  # coarse-grid correction to change the Adam momentum state.

  ##set beta1 to 0.9725
  #for pg in optimizer.param_groups:
  #  pg['betas'] = (0.9725, 0.999)
 
  ## Grab Adam state, and clone it
  #import copy
  #state_dict = copy.deepcopy(optimizer.state_dict())
  
  optimizer.zero_grad()
  e_h = [ -ee for ee in e_h]
  write_params_inplace(model, e_h, grad=True)
  optimizer.step()
  
  ## Reload Adam state
  #optimizer.load_state_dict(state_dict)
 
  ##reset beta1 to 0.9
  #for pg in optimizer.param_groups:
  #  pg['betas'] = (0.9, 0.999)


def tb_simple_weighting(lvl, e_h, x_h, v_h, model, data, target, optimizer, criterion, criterion_kwargs, compose, old_loss, e_dot_gradf, mgopt_printlevel, ls_params):
  '''
  Do no line search (save computation) and simply use the alpha stored in
  ls_params to weight the coarse-grid correction. 
  '''
  try:
    alpha = ls_params['alpha']
  except:
    raise ValueError('tb_simple_weighting requires a ls_params dictionary alpha to be defined (i.e., amount to weight coarse-grid correction') 
  ##
  tensor_list_AXPY(alpha, e_h, 1.0, x_h, inplace=True)
  return alpha


def tb_simple_ls(lvl, e_h, x_h, v_h, model, data, target, optimizer, criterion, criterion_kwargs, compose, old_loss, e_dot_gradf, mgopt_printlevel, ls_params):
  '''
  Simple line-search: Add e_h to fine parameters.  Test five different alpha
  values.  Choose one that best minimizes loss.
  '''
  rank = model.parallel_nn.fwd_app.mpi_comm.Get_rank()
  try:
    alphas = ls_params['alphas']
  except:
    raise ValueError('tb_simple_ls requires a ls_params dictionary alphas defined (i.e., the alphas to test during the line search')


  best_loss = 10**10
  winner = -1

  for aa, alpha in enumerate(alphas):
    # Add error update to x_h:  alpha*e_h + x_h --> x_h
    tensor_list_AXPY(alpha, e_h, 1.0, x_h, inplace=True)

    with torch.enable_grad():
      new_loss = compute_fwd_pass(lvl, model, data, target, criterion, criterion_kwargs, compose, v_h)

    root_print(rank, mgopt_printlevel, 2, "  LS Alpha Test:        " + str(alpha) + "  Loss = " + str(new_loss))

    # Is this a better loss?
    if new_loss < best_loss:
      best_loss = new_loss
      winner = aa

    # Undo error update to x_h:  -alpha*e_h + x_h --> x_h
    tensor_list_AXPY( -alpha, e_h, 1.0, x_h, inplace=True)

  ##
  # end for-loop

  # Choose the winning alpha
  tensor_list_AXPY(alphas[winner], e_h, 1.0, x_h, inplace=True)

  # Print alpha chosen 
  root_print(rank, mgopt_printlevel, 2, "  LS Alpha chosen:        " + str(alphas[winner]) + "  Loss = " + str(best_loss))

  return alphas[winner]


def tb_simple_backtrack_ls(lvl, e_h, x_h, v_h, model, data, target, optimizer, criterion, criterion_kwargs, compose, old_loss, e_dot_gradf, mgopt_printlevel, ls_params):
  '''
  Simple line-search: Add e_h to fine parameters.  If loss has
  diminished, stop.  Else subtract 1/2 of e_h from fine parameters, and
  continue until loss is reduced.
  '''
  rank = model.parallel_nn.fwd_app.mpi_comm.Get_rank()
  try:
    n_line_search = ls_params['n_line_search']
    alpha = ls_params['alpha']
    c1 = ls_params['c1']
  except:
    raise ValueError('tb_simple_backtrack_ls requires a ls_params dictionary with n_line_search, alpha, and c1 (Armijo condition) as dictionary keys.')

  # must be a descent direction
  if e_dot_gradf>=0.0:
    root_print(rank, mgopt_printlevel, 2, 
               f"  LS Alpha used ({lvl}):     {0.0:.4f}  e_dot_gradf:  {e_dot_gradf:.4e} ... canceled...not a descent direction")
    return 0.0

  # Add error update to x_h
  # alpha*e_h + x_h --> x_h
  tensor_list_AXPY(alpha, e_h, 1.0, x_h, inplace=True)

  # Start Line search
  satisfied = False
  for m in range(n_line_search):
    #print("line-search, alpha=", alpha)
    with torch.enable_grad():
      new_loss = compute_fwd_pass(lvl, model, data, target, criterion, criterion_kwargs, compose, v_h)
    
    # Check Wolfe condition (i), also called the Armijo rule
    #  f(x + alpha p) <=  f(x) + c alpha <p, grad f(x) >
    if new_loss < old_loss + c1*alpha*e_dot_gradf:
      satisfied = True
      break
    elif m < (n_line_search-1): 
      # loss is NOT reduced enough, continue line search
      alpha = alpha/2.0
      tensor_list_AXPY(-alpha, e_h, 1.0, x_h, inplace=True)
  ##
  # end for-loop

  # If Wolfe condition (i) never satisfied, subtract off the rest of e_h from x_h, 
  # i.e., change x_h back to what you started with
  if not satisfied:
    tensor_list_AXPY(-alpha, e_h, 1.0, x_h, inplace=True)
    alpha = 0.0

  # Double alpha, and store for next time in ls_params, before returning
  root_print(rank, mgopt_printlevel, 2, 
             f"  LS Alpha used ({lvl}):     {alpha:.4f}  e_dot_gradf:  {e_dot_gradf:.4e}  old_loss + c1*alpha*e_dot_gradf = {old_loss + c1*alpha*e_dot_gradf:.6f}")

  return alpha



####################################################################################
####################################################################################
            
def compute_levels(num_steps,min_coarse_size,cfactor): 
  from math import log, floor 
  # we want to find $L$ such that ( max_L min_coarse_size*cfactor**L <= num_steps)
  levels =  floor(log(num_steps/min_coarse_size,cfactor))+1 

  if levels<1:
    levels = 1
  return levels
# end compute_levels

####################################################################################
####################################################################################
# Small Helper Functions 

def root_print(rank, printlevel_cutoff, importance, s):
  ''' 
  Parallel print routine 
  Only print if rank == 0 and the message is "important"
  '''
  if rank==0:
    if importance <= printlevel_cutoff:
      print(s, flush=True)

def unpack_arg(v):
  ''' Helper function for unpacking arguments '''
  if isinstance(v, tuple):
      return v[0], v[1]
  else:
      return v, {}

def check_has_args(arg_dict, list_to_check, method):
  ''' Check that arg_dict has a key associated with every member of list_to_check '''
  for to_check in list_to_check:
    if not (to_check in arg_dict):
      raise ValueError('Missing arguement for ' + method + ':  ' + to_check)


####################################################################################
####################################################################################



class mgopt_solver:
  """
  Stores the MG/Opt hierarchy of PyTorch (usually TorchBraid) neural networks, and
  implements the MG/Opt cycling for multilevel optimization.   

  Typically, initialize_with_nested_iteration(...) is called to initialize the
  hierarchy with nested iteration, followed by multilevel optimization with
  mgopt_solve(...) 

  Attributes
  ----------
  levels : level array

    Array of level objects that contain the information needed to do
    optimization at that level  and  coarsen/interp parameters and states. See
    levels definition below.

  Methods
  -------
  operator_complexity()
    A measure of the size of the multigrid hierarchy.
  options_used()
    Print out the options used to generate the MG/Opt hierarchy
  initialize_with_nested_iteration()
    Create a hiearchy of neural network models for MG/Opt to use
  mgopt_solve()
    Iteratively solves the optimization problem 

  """

  class level:
    """Stores one level of the multigrid hierarchy.

    Attributes
    ----------
    model           : PyTorch NN model, tested so far with only with TorchBraid ParallelNet model 
    network         : tuple describing the model setup parameters 
    interp_params   : tuple describing the option selected for interpolating network parameters 
    optims          : tuple describing the option selected for the underlying optimizationg method 
    criterions      : tuple describing the option selected for the criterion (objective)
    restrict_params : tuple describing the option selected for restricting network parameters 
    restrict_grads  : tuple describing the option selected for restricting gradients 
    restrict_states : tuple describing the option selected for restricting network states
    interp_states   : tuple describing the option selected for interpolating network states 
    line_search     : tuple describing the option selected for doing a line-search
    optimizer       : PyTorch optimizer for smoothing at each level 
    out_ls_step     : Output diagnostic, the size of the line search step
    """

    def __init__(self):
      """Level construct (empty)."""
      pass


  def __init__(self,device=None):
    """ MG/Oopt constructor """
    self.levels = []

    self.device = device
    if self.device==None:
      self.device = torch.device('cpu')

  def __repr__(self):
    """Print basic statistics about the multigrid hierarchy."""
    model = self.levels[0].model
    comm = model.parallel_nn.fwd_app.mpi_comm
    my_rank = comm.Get_rank()

    # All processors needed to compute op complexity
    (total_op_comp, trainable_op_comp, total_params_per_level, trainable_params_per_level) = self.operator_complexity()

    # Only rank 0 prints output
    if my_rank == 0:
      output = '\nMG/Opt Solver\n'
      output +=f'Device: {self.device}\n'
      output += 'Number of Levels:     %d\n' % len(self.levels)
      output += 'Total Op Complexity: %6.3f\n' % total_op_comp 
      output += 'Trainable Op Complexity: %6.3f\n' % trainable_op_comp 
      
      output += '  level      total params       trainable params \n'
      for n, level in enumerate(self.levels):
        output += '   %2d   %9d [%5.2f%%]   %9d [%5.2f%%] \n' % (n, 
                      total_params_per_level[n], 
                      (100 * float(total_params_per_level[n]) / float(sum(total_params_per_level))), 
                      trainable_params_per_level[n], 
                      (100 * float(trainable_params_per_level[n]) / float(sum(trainable_params_per_level))) )   
      
      return output
    
    else:
      return ""

  def get_total_param_count(self, model):
    """
    Return a length 2 array containing a global count of 
    [ total network params,  trainable network params ]
    
    Result only accurate on rank 0 (reduce is used, not all-reduce)
    """
    comm = model.parallel_nn.fwd_app.mpi_comm
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    my_param_count = np.array( (total_params, trainable_params), dtype=np.float64 )
    total_param_count = np.zeros_like(my_param_count)
    # Use the lower-level comm.Reduce (not comm.reduce)
    comm.Reduce( [my_param_count, MPI.DOUBLE], [total_param_count, MPI.DOUBLE], op=MPI.SUM, root=0)
    return total_param_count

  def operator_complexity(self):
    """Operator complexity of this multigrid hierarchy.

    Returns 4-Tuple
      - Number of total parameters on level 0     / Total number of parameters on all levels 
      - Number of trainable parameters on level 0 / Number of trainable parameters on all levels 
      - Array of the total param count on each level
      - Array of the trainable param count on each level
    """
    model = self.levels[0].model
    comm = model.parallel_nn.fwd_app.mpi_comm
    my_rank = comm.Get_rank()

    total_params_per_level = []
    trainable_params_per_level = []

    for lvl in self.levels:
      model = lvl.model
      total_param_count = self.get_total_param_count(model)  # note that these numbers are only correct on rank 0
      if my_rank == 0:
        total_params_per_level.append(total_param_count[0])
        trainable_params_per_level.append(total_param_count[1])
    
    # Have only rank 0 compute the op complexity for output
    if my_rank == 0:
      total_params_per_level = np.array(total_params_per_level) 
      trainable_params_per_level = np.array(trainable_params_per_level) 
      
      if total_params_per_level.shape[0] > 0:
        total_op_comp =  np.sum(total_params_per_level) / total_params_per_level[0]
        trainable_op_comp =  np.sum(trainable_params_per_level) / trainable_params_per_level[0]
        
        return (total_op_comp, trainable_op_comp, total_params_per_level, trainable_params_per_level)
      
    else:
      return (-1, -1, -1, -1)


  def options_used(self):
    """ Print the options selected to form the hierarchy """
    
    def print_option(o, indent="  ", attr_name=""):
      ''' Helper function to print the user-defined options to a formatted string'''
      method,args = unpack_arg(o)
      output = indent + attr_name + method + '\n' 
      if args == {}:
        output += indent + indent + "Parameters: None\n"
      else:
        for a in args:
          if a != 'model': # 'model' can be a key for some arguments, like criterion, but does not need to be printed
            output += indent + indent + a + " : " + str(args[a]) + '\n'
      ##
      return output

    model = self.levels[0].model
    rank = model.parallel_nn.fwd_app.mpi_comm.Get_rank()
    output = ""
    # Process global parameters
    if hasattr(self, 'preserve_optim'):    output = output + "MG/Opt global parameters\n"
    if hasattr(self, 'nrelax_pre'):    output = output + "  nrelax_pre: " + str(self.nrelax_pre)  + '\n' 
    if hasattr(self, 'nrelax_post'):   output = output + "  nrelax_post: " + str(self.nrelax_post) + '\n' 
    if hasattr(self, 'nrelax_coarse'): output = output + "  nrelax_coarse: " + str(self.nrelax_coarse) + '\n\n' 
    if hasattr(self, 'preserve_optim'): output = output + "  preserve_optim: " + str(self.preserve_optim) + '\n' 
    if hasattr(self, 'zero_init_guess'): output = output + "  zero_init_guess: " + str(self.zero_init_guess) + '\n\n' 
    
    # Process per-level parameters
    for k, lvl in enumerate(self.levels):
      output = output + "MG/Opt parameters from level " + str(k) + '\n'
      if hasattr(self.levels[k], 'network'): output = output + print_option(lvl.network, attr_name="network: ") + '\n' 
      if hasattr(self.levels[k], 'interp_params'): output = output + print_option(lvl.interp_params, attr_name="interp_params: ") + '\n'
      if hasattr(self.levels[k], 'optims'): output = output + print_option(lvl.optims, attr_name="optims: ") + '\n'
      if hasattr(self.levels[k], 'criterions'): output = output + print_option(lvl.criterions, attr_name="criterion: ") + '\n'
      if hasattr(self.levels[k], 'restrict_params'): output = output + print_option(lvl.restrict_params, attr_name="restrict_params: ") + '\n'
      if hasattr(self.levels[k], 'restrict_grads'): output = output + print_option(lvl.restrict_grads, attr_name="restrict_grads: ") + '\n'
      if hasattr(self.levels[k], 'restrict_states'): output = output + print_option(lvl.restrict_states, attr_name="restrict_states: ") + '\n'
      if hasattr(self.levels[k], 'interp_states'): output = output + print_option(lvl.interp_states, attr_name="interp_states: ") + '\n'
      if hasattr(self.levels[k], 'line_search'): output = output + print_option(lvl.line_search, attr_name="line_search: ") + '\n'
    ##
    root_print(rank, 1, 1, output)
      


  def initialize_with_nested_iteration(self, model_factory, ni_steps, 
                                       train_loader, 
                                       test_loader, 
                                       networks, 
                                       epochs           = 1, 
                                       log_interval     = 1, 
                                       mgopt_printlevel = 1,
                                       interp_params    = "tb_get_injection_interp_params", 
                                       optims           = ("pytorch_sgd", { 'lr':0.01, 'momentum':0.9}), 
                                       criterions       = "tb_mgopt_cross_ent", 
                                       preserve_optim   = True,
                                       seed             = None,
                                       zero_init_guess  = False):
    """
    Use nested iteration to create a hierarchy of models

    Parameters
    ----------

    model_factory : lambda (level,**model_args) -> PyTorch Module
      A lambda to construct the model on each level of the hierarchy. The dictonary
      aguments are to be used by the lambda for specific module options 

    ni_steps : array
      array of the number of time_steps at each level of nested iteration, e.g., [1, 2, 4]
      Note: the sequence of steps must be constant refinements, e.g., by a factor of 2 or 3
      Note: length of ni_steps also defines the number of nested iteraiton levels

    train_loader : PyTorch data loader 
      Data loader for training
   
    test_loader  : PyTorch data loader 
      Data loader for testing
    
    networks : list
      networks[k] describes the network architecture at level k in the nested
      iteration hierarchy, starting from fine to coarse, with level k=0 the finest. 
      Note: the architecture at level k (networks[k]) will typically state the number of layers at level k
      Note: this number of layers at level k must be consistent with the ni_steps array
      
    epochs : int
      Number of training epochs
    
    log_interval : int
      How often to output batch-level timing and loss data

    mgopt_printlevel : int
      output level for mgopt.  0 = no output, 1 = some output, 2 = detailed output

    interp_params : list|string|tuple
      interp_params[k] describes how to interpolate the network parameters at
      level k in the nested iteration hierarchy, with level k=0 the finest.
      -> If string or tuple, then the string/tuple defines option at all levels.

    optims : list|string|tuple
      optims[k] describes the optimization strategy to use at level k in the
      nested iteration hierarchy, with level k=0 the finest.  
      -> If string or tuple, then the string/tuple defines option at all levels.
   
    criterions : list|string|tuple
      criterions[k] describes the criterion or objective function at level k
      in the nested iteration hierarchy, with level k=0 the finest.
      -> If string or tuple, then the string/tuple defines option at all levels.

      Note: If writing a new criterion, it needs to support two modes.  The
      classic criterion(output, target), and a mode that supports the
      additional MG/Opt term, criterion(output, target, x_h, v_h)

    preserve_optim : boolean
      Default True.  If True, preserve the optimizer state between batches.  If
      False, reset optimizer state always before a step.

    seed : int
      seed for random number generate (e.g., when initializing weights)

    zero_init_guess : int
      If 1, then initialize nested iteration with all zero parameters on
      initial level.  Useful for parallel reproducibility.  Default is 0,False.
  

    Notes
    -----
    The list entries above are desiged to be in a variety of formats.
    If entry is 'string', then the 'string' corresponds to a parameter option 
      to use at all levels.
    If entry is tuple of ('string', param_dict), then string is a supported
      parameter option that takes parameters 'param_dict'
    If a list, the entry [k] is a 'string' or tuple defining the option at 
      level k, which k=0 the finest.


    Returns
    -------
    Initialized hierarchy for MG/Opt 
    No direct output. Changes done internally to multilevel solver object

    """
          
    nlevels = len(ni_steps)
    rank = MPI.COMM_WORLD.Get_rank()

    ##
    # Process arguments 
    interp_params = self.levelize_argument(interp_params, nlevels)
    optims        = self.levelize_argument(optims, nlevels)
    criterions    = self.levelize_argument(criterions, nlevels)
    if( len(ni_steps) != len(networks) ):
      raise ValueError('Length of ni_steps must equal length of networks, i.e., you must have a network architecture defined for each level of nested iteration')
    
    ##
    # Reverse order of arguments, because we start from coarse to fine
    interp_params.reverse()
    optims.reverse()
    criterions.reverse()
    networks.reverse()
    
    ##
    # Seed the generator for the below training 
    if seed is not None:
      torch.manual_seed(torchbraid.utils.seed_from_rank(seed, rank))
    
    ##
    # Check ni_steps that it is a constant r_factor
    if nlevels > 1:
      ni_rfactor = int(max(ni_steps[1:] / ni_steps[:-1]))
      ni_rfactor_min = int(min(ni_steps[1:] / ni_steps[:-1]))
    else:
      ni_rfactor = -1
      ni_rfactor_min = -1
    #
    self.ni_steps = ni_steps
    self.ni_rfactor = ni_rfactor
    if( ni_rfactor != ni_rfactor_min):
      raise ValueError('Nested iteration (ni_steps) should use a single constant refinement factor') 
    #  
    self.preserve_optim = bool(preserve_optim)
    self.zero_init_guess = bool(zero_init_guess)

    ##
    # Initialize self.levels with nested iteration
    for k, steps in enumerate(ni_steps):
      
      ##
      # Create new model for this level
      model_name, kwargs = unpack_arg(networks[k])
      if model_name=='Factory':
        model = model_factory(k,**kwargs) # user lambda that takes the arugments and builds the network
        model = model.to(self.device)
        model.parallel_nn.setBwdStorage(0)  # Only really needed if Braid will create a single time-level.  
      else:
        raise ValueError('Unsupported model: ' + model_string)
      
      ##
      # Get rank from model
      if k == 0: rank = model.parallel_nn.fwd_app.mpi_comm.Get_rank()

      ##
      # For parallel reproducibility, set all parameters to 0
      if self.zero_init_guess:
        with torch.no_grad():
          x_h = get_params(model, deep_copy=False, grad=False)
          for param in x_h: param[:] = 0.0

      ##
      # Create optimizer, if preserving optimizer state between batches
      if self.preserve_optim:
        (optimizer, optim_kwargs) = self.process_optimizer(optims[k], model)

      ##
      # Select Criterion (objective) and compose function
      (criterion, compose, criterion_kwargs) = self.process_criterion(criterions[k], model)

      ##
      # First, we must allocate everything in Braid with a dummy iteration
      # - For interpolation if on a refined level (levels > 0), or 
      # - On every level, if doing epochs=0, here NI is only allocating the hierarchy 
      sys.stdout.flush()
      if (len(self.levels) > 0) or (epochs == 0): 
        model.eval()
        for (data, target) in train_loader:
          data = data.to(self.device)
          target = target.to(self.device)
          output = model(data)
          loss = compose(criterion, output, target, **criterion_kwargs)
          loss.backward()
          break

      ##
      # Select Interpolate weights from coarser model to the new model
      if (len(self.levels) > 0):
        (get_interp_params, interp_params_kwargs) = self.process_get_interp_params(interp_params[k])
        new_params = get_interp_params(model, self.levels[-1].model, **interp_params_kwargs)
        write_params_inplace(model, new_params)
        del new_params

      ##
      # Diagnostic printing
      total_param_count = self.get_total_param_count(model)  # note that these numbers are only correct on rank 0
      if rank == 0:
        root_print(rank, mgopt_printlevel, 1, '\nNested Iter Level:  ' + str(k) )
        root_print(rank, mgopt_printlevel, 1, '  optimizing %d steps' % steps)
        root_print(rank, mgopt_printlevel, 1, '  total params: %d' % total_param_count[0])
        root_print(rank, mgopt_printlevel, 1, '  train params: %d' % total_param_count[1])
        root_print(rank, mgopt_printlevel, 1, '')

      
      ##
      # Begin epoch loop
      epoch_times = []
      test_times = []
      for epoch in range(1, epochs + 1):
        start_time = timer()
        # call train_epoch, depending on whether the optimizer state is preserved between runs
        if self.preserve_optim:
          train_epoch(rank, model, train_loader, optimizer, epoch, criterion, criterion_kwargs, compose, log_interval, self.device, mgopt_printlevel)
        else:
          train_epoch(rank, model, train_loader, optims[k], epoch, criterion, criterion_kwargs, compose, log_interval, self.device, mgopt_printlevel)
        end_time = timer()
        epoch_times.append( end_time-start_time )
    
        # test() is designed to be general for PyTorch networks
        start_time = timer()
        test(rank, model, test_loader, criterion, criterion_kwargs, compose, self.device, mgopt_printlevel, indent='\n  ')
        end_time = timer()
        test_times.append( end_time-start_time )
      
      ##
      # Store model and parameters
      self.levels.append(self.level())
      self.levels[-1].model = model
      self.levels[-1].network = networks[k]
      self.levels[-1].interp_params = interp_params[k]
      self.levels[-1].optims = optims[k]
      self.levels[-1].criterions = criterions[k]
      self.levels[-1].out_ls_step = []
      if self.preserve_optim:
        self.levels[-1].optimizer = optimizer

      # Print epoch-level time averages
      if epochs == 0:
        root_print(rank, mgopt_printlevel, 1, '  Zero Nested Iteration Epochs -- Only Initializing Hierarchy ')
      elif epochs == 1:
        root_print(rank, mgopt_printlevel, 1, '  Time per epoch: %.2e ' % (stats.mean(epoch_times)) ) 
        root_print(rank, mgopt_printlevel, 1, '  Time per test:  %.2e ' % (stats.mean(test_times)) )
      else:
        root_print(rank, mgopt_printlevel, 1, '  Time per epoch: %.2e (1 std dev %.2e)' % (stats.mean(epoch_times), stats.stdev(epoch_times)))
        root_print(rank, mgopt_printlevel, 1, '  Time per test:  %.2e (1 std dev %.2e)' % (stats.mean(test_times), stats.stdev(test_times)))


    ##
    # Reverse order, so finest level is level 0 
    self.levels.reverse()
    


  def mgopt_solve(self, train_loader, 
                        test_loader, 
                        epochs = 1, 
                        log_interval = 1,
                        mgopt_tol = 0, 
                        mgopt_iter = 1, 
                        nrelax_pre = 1, 
                        nrelax_post = 1,
                        nrelax_coarse = 5, 
                        mgopt_printlevel = 1, 
                        mgopt_levels = None,
                        preserve_optim   = True,
                        restrict_params = ("tb_get_injection_restrict_params", {'grad' : False}), 
                        restrict_grads = ("tb_get_injection_restrict_params", {'grad' : True}), 
                        restrict_states = "tb_injection_restrict_network_state",
                        interp_states = "tb_injection_interp_network_state", 
                        line_search = ('tb_simple_ls', {'alphas' : [0.001, 0.01, 0.1, 0.5, 1.0]})
                        ):    
    """
    Use nested iteration to create a hierarchy of models

    Parameters
    ----------
    
    train_loader : PyTorch data loader 
      Data loader for training
   
    test_loader  : PyTorch data loader 
      Data loader for testing
    
    epochs : int
      Number of training epochs
    
    log_interval : int
      How often to output batch-level timing and loss data
    
    mgopt_tol : float
      If the objective evaluation (loss) drops below this value, then halt 

    mgopt_iter : int
      Number of MG/Opt iterations for each training batch.
      This is an inner loop in the total process.

    nrelax_pre : int
      Number of pre-relaxation steps

    nrelax_post : int
      Number of post-elaxation steps
  
    nrelax_coarse : int
      Number of relaxation steps used to solve the coarse-level 

    mgopt_printlevel : int
      output level for mgopt.  0 = no output, 1 = some output, 2 = detailed output
      ==> Note, turning this option to 2, results in more frequent measurements
          of the loss, and will change the returned loss values

    mgopt_levels : int or None
      Defines number of MG/Opt levels to use.  Must be less than or equal
      to the total number of hierarchy levels.  If None, all levels are used.

    preserve_optim : boolean
      Default True.  If True, preserve the optimizer state between levels and
      between batches.  If False, reset optimizer state always before a step.

    restrict_params : list|string|tuple
      restrict_params[k] describes the strategy for restricting network
      parameters on level k in the MG/Opt hierarchy 
      -> If string or tuple, then the string/tuple defines option at all levels.

    restrict_grads : list|string|tuple
      restrict_grads[k] describes the strategy for restricting network
      gradients on level k in the MG/Opt hierarchy 
      -> If string or tuple, then the string/tuple defines option at all levels.

    restrict_states : list|string|tuple
      restrict_states[k] describes the strategy for restricting network
      states on level k in the MG/Opt hierarchy 
      -> If string or tuple, then the string/tuple defines option at all levels.

    interp_states : list|string|tuple
      interpt_states[k] describes the strategy for interpolating network
      states on level k in the MG/Opt hierarchy 
      -> If string or tuple, then the string/tuple defines option at all levels.

    line_search : list|string|tuple
      line_search[k] describes the strategy for line search with the
      coarse-grid correction on level k in the MG/Opt hierarchy 
      -> If string or tuple, then the string/tuple defines option at all levels.

    Notes
    -----
    The list entries above are desiged to be in a variety of formats.
    If entry is 'string', then the 'string' corresponds to a parameter option 
      to use at all levels.
    If entry is tuple of ('string', param_dict), then string is a supported
      parameter option that takes parameters 'param_dict'
    If a list, the entry [k] is a 'string' or tuple defining the option at 
      level k, which k=0 the finest.

    In general, the restrict/interp state functions work like
      restrict(model_fine, model_coarse, **kwargs)
      interp(model_fine, model_coarse, **kwargs)

    In general, the get_restrict/interp parameter and gradient functions work like
      restricted_params   = get_restrict_params(model_fine, model_coarse, **kwargs)
      restricted_grad     = get_restrict_grad(model_fine, model_coarse, **kwargs)
      interpolated_params = get_interp_params(model_fine, model_coarse, **kwargs)

    
    Returns
    -------
    Trains the hiearchy in place.  Look in self.levels[i].model for the
    trained model on level i, with i=0 the finest level. 
    
    List of loss values from each MG/Opt epoch

    """
    
    model = self.levels[0].model
    rank = model.parallel_nn.fwd_app.mpi_comm.Get_rank()

    ##
    # Store global solve parameters
    self.nrelax_pre = nrelax_pre
    self.nrelax_post = nrelax_post
    self.nrelax_coarse = nrelax_coarse

    ##
    # Determine number of mgopt_levels
    if mgopt_levels == None:
      mgopt_levels = len(self.levels)
    elif mgopt_levels > len(self.levels):
      raise ValueError('Number of mgopt_levels must be less than or equal to the total number of hierarchy levels: ' + str(len(self.levels))) 
    
    ##
    # Process arguments 
    restrict_params  = self.levelize_argument(restrict_params, mgopt_levels)
    restrict_grads   = self.levelize_argument(restrict_grads, mgopt_levels)
    restrict_states  = self.levelize_argument(restrict_states, mgopt_levels)
    interp_states    = self.levelize_argument(interp_states, mgopt_levels)
    line_search      = self.levelize_argument(line_search, mgopt_levels)
        
    ##
    # For epoch-level accuracy measurements, select (i) criterion (objective) for level 0 and (ii) the compose function
    (criterion, compose, criterion_kwargs) = self.process_criterion(self.levels[0].criterions, self.levels[0].model)
    
    ##
    # If preserving the optimizer state, initialize once here (unless NI has already initialized it).
    self.preserve_optim = bool(preserve_optim)
    if self.preserve_optim:
      for k in range(mgopt_levels):
        # Only generate a new optimizer, if one isn't already stored
        if not hasattr(self.levels[k], 'optimizer'):
          (optimizer, optim_kwargs) = self.process_optimizer(self.levels[k].optims, self.levels[k].model)
          self.levels[k].optimizer = optimizer

    ##
    # Begin loop over epochs
    losses = []
    epoch_times = []
    test_times = []
    for epoch in range(1, epochs + 1):
      epoch_time_start = timer()
      
      ##
      # Begin loop over batches
      batch_total_time = 0.0
      for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(self.device)
        target = target.to(self.device)
        ##
        # Initiate recursive MG/Opt solve
        lvl = 0      
        v_h = None
        root_print(rank, mgopt_printlevel, 2, "\nBatch:  " + str(batch_idx)) 
        start_batch_time = timer()

        ##
        # Begin Cycle loop
        for it in range(mgopt_iter):
          root_print(rank, mgopt_printlevel, 2, "MG/Opt Iter:  " + str(it)) 
          loss_item = self.__solve(lvl, data, target, v_h, mgopt_printlevel, 
                                   mgopt_levels, restrict_params, restrict_grads, 
                                   restrict_states, interp_states, line_search)
        ##
        # End cycle loop
        
        losses.append(loss_item)
        end_batch_time = timer()
        batch_total_time += (end_batch_time - start_batch_time)
        
        ##
        # Batch-level diagnostic printing
        if (batch_idx % log_interval == 0) or (batch_idx == (len(train_loader)-1) ):  
          if batch_idx==0:
            root_print(rank, mgopt_printlevel, 1, '')

          root_print(rank, mgopt_printlevel, 2, "\n------------------------------------------------------------------------------")
          root_print(rank, mgopt_printlevel, 1, '  Train Epoch: {} [{}/{} ({:.0f}%)]     \tLoss: {:.9f}\tTime Per Batch {:.6f}'.format(
              epoch, (batch_idx+1) * len(data), len(train_loader.dataset),
              100. * (batch_idx+1) / len(train_loader), losses[-1], batch_total_time/(batch_idx+1.0)))
          root_print(rank, mgopt_printlevel, 2, "------------------------------------------------------------------------------")

      ##
      # End Batch loop
      epoch_time_end = timer()
      epoch_times.append( epoch_time_end - epoch_time_start)

      ##
      # Measure training accuracy
      start = timer()
      test(rank, self.levels[0].model, test_loader, criterion, criterion_kwargs, compose, self.device, mgopt_printlevel, indent='\n  ')
      end = timer()
      test_times.append( end -start )
      
      ##
      # For printlevel 2, also test accuracy ong coarse-levels
      if(mgopt_printlevel >= 2):
        for i, level in enumerate(self.levels[1:]):
          root_print(rank, mgopt_printlevel, 2, '  Test accuracy information for level ' + str(i+1))
          test(rank, level.model, test_loader, criterion, criterion_kwargs, compose, self.device, mgopt_printlevel, indent='    ')
      
      ##
      # Print epoch-level time averages
      if epoch == 1:
        root_print(rank, mgopt_printlevel, 1, '  Time per epoch: %.2e ' % (stats.mean(epoch_times)) ) 
        root_print(rank, mgopt_printlevel, 1, '  Time per test:  %.2e ' % (stats.mean(test_times)) )
      else:
        root_print(rank, mgopt_printlevel, 1, '  Time per epoch: %.2e (1 std dev %.2e)' % (stats.mean(epoch_times), stats.stdev(epoch_times)))
        root_print(rank, mgopt_printlevel, 1, '  Time per test:  %.2e (1 std dev %.2e)' % (stats.mean(test_times), stats.stdev(test_times)))


      if losses[-1] < mgopt_tol:
        break

    ##
    # End while loop

    return losses


  def __solve(self, lvl, data, target, v_h, mgopt_printlevel, 
          mgopt_levels, restrict_params, restrict_grads, 
          restrict_states, interp_states, line_search):

    """
    Recursive solve function for carrying out one MG/Opt iteration
    See solve() for parameter description
    """
    
    ##
    # Grab fine and coarse models and optimizers
    # We regenerate the optimizer each time, as some optimizers store state
    model = self.levels[lvl].model
    coarse_model = self.levels[lvl+1].model
    comm = model.parallel_nn.fwd_app.mpi_comm
    rank = comm.Get_rank()
    # If preserving the optimizer state, grab existing optimizer from hierarchy.  Else, generate new one.
    if self.preserve_optim:
      optimizer = self.levels[lvl].optimizer
      coarse_optimizer = self.levels[lvl+1].optimizer # If a clean coarse-grid correction with no momentum is desired, comment this out
    else:
      (optimizer, optim_kwargs) = self.process_optimizer(self.levels[lvl].optims, self.levels[lvl].model)
      (coarse_optimizer, coarse_optim_kwargs) = self.process_optimizer(self.levels[lvl+1].optims, self.levels[lvl+1].model)
    ##
    root_print(rank, mgopt_printlevel, 2, "\n  Level:  " + str(lvl)) 


    ##
    # Store new options needed by MG/Opt
    self.levels[lvl].restrict_params = restrict_params[lvl]
    self.levels[lvl].restrict_grads  = restrict_grads[lvl]
    self.levels[lvl].restrict_states = restrict_states[lvl]
    self.levels[lvl].interp_states   = interp_states[lvl]
    self.levels[lvl].line_search     = line_search[lvl]
    # Rembember these, options were stored previously by nested iteration
    #   self.levels[lvl].interp_params
    #   self.levels[lvl].optims
    #   self.levels[lvl].criterions

    ##
    # Process the user-specifiec options for restriction, interpolation, criterion, ...
    (criterion, compose, criterion_kwargs)        = self.process_criterion(self.levels[lvl].criterions, model)
    (do_line_search, ls_kwargs)                   = self.process_line_search(self.levels[lvl].line_search)
    (do_restrict_states, restrict_states_kwargs)  = self.process_restrict_states(self.levels[lvl].restrict_states)
    (get_restrict_params, restrict_params_kwargs) = self.process_get_restrict_params(self.levels[lvl].restrict_params)
    (get_restrict_grad, restrict_grad_kwargs)     = self.process_get_restrict_grad(self.levels[lvl].restrict_grads)
    (get_interp_params, interp_params_kwargs)     = self.process_get_interp_params(self.levels[lvl].interp_params)
    (do_interp_states, interp_states_kwargs)      = self.process_interp_states(self.levels[lvl].interp_states)
    #
    (coarse_criterion, coarse_compose, coarse_criterion_kwargs) = self.process_criterion(self.levels[lvl+1].criterions, coarse_model)
    
    ##
    # Fixed-point test level output
    if mgopt_printlevel == 3:
      x_h = get_params(model, deep_copy=False, grad=False)
      root_print(rank, mgopt_printlevel, 3, "  Pre-MG/Opt solution norm:       " + str(tensor_list_dot(x_h, x_h, comm).item()) ) 
    
    # 1. relax (carry out optimization steps)
    for k in range(self.nrelax_pre):
      loss_scalar = compute_fwd_bwd_pass(lvl, optimizer, model, data, target, criterion, criterion_kwargs, compose, v_h)
      root_print(rank, mgopt_printlevel, 2, "  Pre-relax loss:       " + str(loss_scalar) ) 
      optimizer.step()

    # 2. compute new gradient g_h
    # First evaluate network, forward and backward.  Return value is scalar value of the loss.
    fine_loss = compute_fwd_bwd_pass(lvl, optimizer, model, data, target, criterion, criterion_kwargs, compose, v_h)
    root_print(rank, mgopt_printlevel, 2, "  Pre-relax done loss:  " + str(fine_loss)) 
    with torch.no_grad():
      g_h = get_params(model, deep_copy=True, grad=True)
      
    ## 
    # We do not understand the stocasticity yet...but when we do, could use
    # momentum as g_h.
    ##if len(optimizer.state) > 0:
    ##  g_h = get_adam_momentum(model, optimizer)
    ##  # Write g_h inplace here to model, for restriction in step 3 
    ##  write_params_inplace(model, g_h, grad=True)
    ##else:
    ##  g_h = get_params(model, deep_copy=True, grad=True)

    # 3. Restrict 
    #    (i)   Network state (primal and adjoint), 
    #    (ii)  Parameters (x_h), and 
    #    (iii) Gradient (g_h) to H
    # x_H^{zero} = R(x_h)   and    \tilde{g}_H = R(g_h)
    with torch.no_grad():
      #do_restrict_states(model, coarse_model, **restrict_states_kwargs)
      gtilde_H = get_restrict_grad(model, coarse_model, **restrict_grad_kwargs)
      x_H      = get_restrict_params(model, coarse_model, **restrict_params_kwargs) 
      # For x_H to take effect, these parameters must be written to the next coarser network
      write_params_inplace(coarse_model, x_H)
      # Must store x_H for later error computation
      x_H_zero = tensor_list_deep_copy(x_H)


    ## 
    # We do not understand the stocasticity yet...but when we do, could consider
    # restricting Adam state to coarse levels
    ##tb_injection_restrict_adam_state(model, coarse_model, optimizer, coarse_optimizer, 2)


    # 4. compute gradient on coarse level, using restricted parameters
    #  g_H = grad( f_H(x_H) )
    #
    # Evaluate gradient.  For computing fwd_bwd_pass, give 0 as first
    # parameter, so that the MG/Opt term is turned-off.  We just want hte
    # gradient of f_H here.
    loss_scalar = compute_fwd_bwd_pass(0, coarse_optimizer, coarse_model, data, target, coarse_criterion, coarse_criterion_kwargs, coarse_compose, None)
    with torch.no_grad():
      g_H = get_params(coarse_model, deep_copy=True, grad=True)
      
      ##
      # We do not understand the stocasticity yet...but when we do, could
      # consider using a coarse-grid expected gradient for g_H
      ##if len(coarse_optimizer.state) > 0:
      ##  g_H = get_adam_momentum(coarse_model, coarse_optimizer)
      ##else:
      ##  g_H = get_params(coarse_model, deep_copy=True, grad=True)

    # 5. compute coupling term
    #  v = g_H - \tilde{g}_H
    with torch.no_grad():
      v_H = tensor_list_AXPY(1.0, g_H, -1.0, gtilde_H)

    # 6. solve coarse-grid 
    #  x_H = min f_H(x_H) - <v_H, x_H>
    if (lvl+2) == mgopt_levels:
      # If on coarsest level, do a "solve" by carrying out a number of relaxation steps
      root_print(rank, mgopt_printlevel, 2, "\n  Level:  " + str(lvl+1))
      for m in range(self.nrelax_coarse):
        loss_scalar_coarse = compute_fwd_bwd_pass(lvl+1, coarse_optimizer, coarse_model, data, target, coarse_criterion, coarse_criterion_kwargs, coarse_compose, v_H)
        coarse_optimizer.step()
        root_print(rank, mgopt_printlevel, 2, "  Coarsest grid solve loss: " + str(loss_scalar_coarse)) 
    else:
      # Recursive call
      self.__solve(lvl+1, data, target, v_H, mgopt_printlevel, 
                   mgopt_levels, restrict_params, restrict_grads,
                   restrict_states, interp_states, line_search)
    #
    root_print(rank, mgopt_printlevel, 2, "  Recursion exited\n")
      
    # 7. Interpolate 
    #    (i)  error correction to fine-level, and 
    #    (ii) network state to fine-level (primal and adjoint)
    #  e_h = P( x_H - x_H^{init})
    with torch.no_grad():
      x_H = get_params(coarse_model, deep_copy=False, grad=False)
      e_H = tensor_list_AXPY(1.0, x_H, -1.0, x_H_zero)
      #
      # to correctly interpolate e_H --> e_h, we need to put these values in a
      # network, so the interpolation knows the layer-parallel structure and
      # where to refine.
      write_params_inplace(coarse_model, e_H)
      e_h = get_interp_params(model, coarse_model, **interp_params_kwargs) 
      #do_interp_states(model, coarse_model, **interp_states_kwargs)
      root_print(rank, mgopt_printlevel, 2, "  Norm of error correction:       " + str(np.sqrt(tensor_list_dot(e_h, e_h, comm).item())) ) 
      

    # 8. apply linesearch to update x_h
    #  x_h = x_h + alpha*e_h
    with torch.no_grad():
      x_h = get_params(model, deep_copy=False, grad=False)
      e_dot_gradf = tensor_list_dot(e_h, g_h, comm).item()
      # Note, that line_search can store values between runs, like alpha, by having ls_kwargs = { 'ls_params' : {'alpha' : ...}} 
      ls_alpha = do_line_search(lvl, e_h, x_h, v_h, model, data, target, optimizer, criterion, criterion_kwargs, compose, fine_loss, e_dot_gradf, mgopt_printlevel, **ls_kwargs)
      self.levels[lvl].out_ls_step += [ls_alpha]


    # 9. post-relaxation
    for k in range(self.nrelax_post):
      loss_scalar = compute_fwd_bwd_pass(lvl, optimizer, model, data, target, criterion, criterion_kwargs, compose, v_h)
      if (k==0):  root_print(rank, mgopt_printlevel, 2, "  CG Corr done loss:    " + str(loss_scalar) ) 
      else:       root_print(rank, mgopt_printlevel, 2, "  Post-relax loss:      " + str(loss_scalar) )
      optimizer.step()
    ##
    if (mgopt_printlevel == 2):  
      loss_scalar = compute_fwd_pass(lvl, model, data, target, criterion, criterion_kwargs, compose, v_h)
      root_print(rank, mgopt_printlevel, 2, "  Post-relax loss:      " + str(loss_scalar) )
  
    ##
    # Fixed-point test level output
    if mgopt_printlevel == 3:
      root_print(rank, mgopt_printlevel, 3, "  Post-MG/Opt solution norm:       " + str(tensor_list_dot(x_h, x_h, comm).item()) ) 


    return loss_scalar   
        
        

  def levelize_argument(self, to_levelize, max_levels):
        """
        Helper function to preprocess parameter specifications so that they
        become a a per-level list specifying the paramter on that level.

        Parameters
        ----------
        to_levelize : {string, tuple, list}
            Parameter to preprocess, i.e., levelize and convert to a level-by-level
            list such that entry i specifies the parameter at level i
        max_levels : int
            Defines the maximum number of levels considered

        Returns
        -------
        to_levelize : list
            The parameter list such that entry i specifies the parameter choice
            at level i.

        Notes
        --------
        This routine is needed because the user will pass in a parameter option
        such as arg1='something' or arg1=['something1, something2] or
        arg1=('something, {}), and we want these formats to all be converted to 
        per-level parameter specifications.

        All tuple arguments are assumed to be of the form 
        (string, dict)

        """
        if isinstance(to_levelize, tuple):
            to_levelize = [(to_levelize[0], to_levelize[1].copy()) for i in range(max_levels)]
        elif isinstance(to_levelize, str):
            to_levelize = [(to_levelize, {}) for i in range(max_levels)]
        elif isinstance(to_levelize, list):
            if len(to_levelize) < max_levels:
                mlz = max_levels - len(to_levelize)
                toext = [(to_levelize[-1][0], to_levelize[-1][1].copy()) for i in range(mlz)]
                to_levelize.extend(toext)
        elif to_levelize is None:
            to_levelize = [(None, {}) for i in range(max_levels)]

        return to_levelize

  ###
  # Begin user option processing routines
  ###

  def process_criterion(self, option, model):
    ''' Return Criterion (objective) and the compose function for option '''
    method, criterion_kwargs = unpack_arg(option)
    criterion_kwargs['model'] = model   # some criteria (Loss functions) need the model to compute the loss

    if method == "tb_mgopt_cross_ent":
      criterion = tb_mgopt_cross_ent
      compose = model.compose
    elif method == "tb_mgopt_cross_ent_plus_continuity":
      criterion = tb_mgopt_cross_ent_plus_continuity
      compose = model.compose
    elif method == "tb_mgopt_regression":
      criterion = tb_mgopt_regression
      compose = model.compose
    else:
      raise ValueError('Unsupported criterion: ' + method)  
    ##
    return criterion, compose, criterion_kwargs


  def process_optimizer(self, option, model):
    ''' Return optimizer for option '''
    method, optim_kwargs = unpack_arg(option)
    if method == "pytorch_sgd":
      optimizer = optim.SGD(model.parameters(), **optim_kwargs)
    elif method == "pytorch_adam":
      optimizer = optim.Adam(model.parameters(), **optim_kwargs)
    else:
      raise ValueError('Unsupported optimizer: ' + method)
    ##
    return optimizer, optim_kwargs


  def process_line_search(self, option):
    ''' Return line search for option '''
    method, ls_kwargs = unpack_arg(option)
    if method == "tb_simple_backtrack_ls":
      check_has_args(ls_kwargs, ['ls_params'], method)
      line_search = tb_simple_backtrack_ls
    elif method == "tb_simple_ls":
      check_has_args(ls_kwargs, ['ls_params'], method)
      line_search = tb_simple_ls
    elif method == "tb_simple_weighting":
      check_has_args(ls_kwargs, ['ls_params'], method)
      line_search = tb_simple_weighting
    elif method == "tb_adam_no_ls":
      check_has_args(ls_kwargs, ['ls_params'], method)
      line_search = tb_adam_no_ls
    else:
      raise ValueError('Unsupported line search: ' + method)
    ##
    return line_search, ls_kwargs

   
  def process_restrict_states(self, option):
    ''' Return restrict states for option '''
    method, restrict_states_kwargs = unpack_arg(option)
    if method == "tb_injection_restrict_network_state":
      restrict_states = tb_injection_restrict_network_state
      restrict_states_kwargs.update({'cf' : self.ni_rfactor})
    else:
      raise ValueError('Unsupported restrict state: ' + method)  
    ##
    return restrict_states, restrict_states_kwargs
    

  def process_get_restrict_params(self, option):
    ''' Return restrict params for option '''
    method, restrict_params_kwargs = unpack_arg(option)
    if method == "tb_get_injection_restrict_params":
      get_restrict_params = tb_get_injection_restrict_params
      restrict_params_kwargs.update({'cf' : self.ni_rfactor, 'deep_copy' : True, 'grad' : False})
    elif method == "tb_get_linear_restrict_params":
      get_restrict_params = tb_get_linear_restrict_params
      restrict_params_kwargs.update({'cf' : self.ni_rfactor, 'deep_copy' : True, 'grad' : False})
    elif method == "tb_parallel_get_injection_restrict_params":
      get_restrict_params = tb_parallel_get_injection_restrict_params
      restrict_params_kwargs.update({'cf' : self.ni_rfactor, 'deep_copy' : True, 'grad' : False})
    else:
      raise ValueError('Unsupported restrict params: ' + method)  
    ##
    return get_restrict_params, restrict_params_kwargs
    

  def process_get_restrict_grad(self, option):
    ''' Return restrict grad for option '''
    method, restrict_grad_kwargs = unpack_arg(option)
    if method == "tb_get_injection_restrict_params":
      get_restrict_grad = tb_get_injection_restrict_params
      restrict_grad_kwargs.update({'cf' : self.ni_rfactor, 'deep_copy' : True, 'grad' : True})
    elif method == "tb_get_linear_restrict_params":
      get_restrict_grad = tb_get_linear_restrict_params
      restrict_grad_kwargs.update({'cf' : self.ni_rfactor, 'deep_copy' : True, 'grad' : True})
    elif method == "tb_parallel_get_injection_restrict_params":
      get_restrict_grad = tb_parallel_get_injection_restrict_params
      restrict_grad_kwargs.update({'cf' : self.ni_rfactor, 'deep_copy' : True, 'grad' : True})
    else:
      raise ValueError('Unsupported restrict grad: ' + method)  
    ##
    return get_restrict_grad, restrict_grad_kwargs
    

  def process_get_interp_params(self, option):
    ''' Return interp params for option '''
    method, interp_params_kwargs = unpack_arg(option)
    if method == "tb_get_injection_interp_params":
      get_interp_params = tb_get_injection_interp_params
      interp_params_kwargs.update({'deep_copy' : True, 'grad' : False, 'cf' : self.ni_rfactor})
    elif method == "tb_get_linear_interp_params":
      get_interp_params = tb_get_linear_interp_params
      interp_params_kwargs.update({'deep_copy' : True, 'grad' : False, 'cf' : self.ni_rfactor})
    elif method == "tb_parallel_get_injection_interp_params":
      get_interp_params = tb_parallel_get_injection_interp_params
      interp_params_kwargs.update({'deep_copy' : True, 'grad' : False, 'cf' : self.ni_rfactor})
    else:
      raise ValueError('Unsupported interp params: ' + method)
    ##
    return get_interp_params, interp_params_kwargs
    

  def process_interp_states(self, option):
    ''' Return interp state for option '''
    method, interp_states_kwargs = unpack_arg(option)
    if method == "tb_injection_interp_network_state":
      interp_states = tb_injection_interp_network_state
      interp_states_kwargs.update({'cf' : self.ni_rfactor})
    else:
      raise ValueError('Unsupported interp state: ' + method)
    ##
    return interp_states, interp_states_kwargs
      

