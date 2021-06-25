"""
This file contains:
  - Generic multilevel solver.
    Based on PyAMG multilevel solver, https://github.com/pyamg/pyamg
    PyAMG is released under the MIT license.
  - MG/Opt implementations of the multilevel solver
"""


from __future__ import print_function
from warnings import warn

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


# Second, move the MGOPT stuff
#  Initially two-level
#  MAYBE first add all the params that you'll need in main_mgopt2 and in the function dec
#  MAKE sure all the "extra" niceties of NI carry on down here, like storing params, comments, ...
#  Make sure two codes give the same result
#
#
# Clean up excess code in both files, especially main_mgopt2.py
# CAN you put all torchbraid functions in a file -- separate them somehow?
# ==> Fill out comments below, e.g., what's in a level  and   update the parameters print function for everything added in MGOpt
# Commit
#
# Incorporate all the local learning parameters in args -- have to make sure all the new command line args filter down and affect things, and 
# DO Multilevel
#
# Test code and extend code 
#    - Can you do a fixed point test?  Or load a fully trained network, see if its almost a fixed point?
#    - How to structure mini-batch loops, e.g., one loop outside of MG/Opt and/or one loop inside MG/Opt (used by each relaxation step)
#      Perhaps, you want train_loader to be post-processed into doubly nested
#      list, where each inside list is the set of batch(es) for relaxation to
#       iterate over, [  [],   [], [],  .... ]
# 
#

# Future Work:
#  - Parallel
#  - More control over how the trainloader is iterated  ... inside or outside?
#  - Coarsegrid convexity tweak from 


__all__ = [ 'mgopt_solver', 'parse_args' ]

####################################################################################
####################################################################################
# Classes and functions that define the basic network types for MG/Opt and TorchBraid.
# Could go into separate file

class OpenLayer(nn.Module):
  def __init__(self,channels):
    super(OpenLayer, self).__init__()
    ker_width = 3
    self.conv = nn.Conv2d(3,channels,ker_width,padding=1)

  def forward(self, x):
    return F.relu(self.conv(x))
# end OpenLayer

class CloseLayer(nn.Module):
  def __init__(self,channels):
    super(CloseLayer, self).__init__()
    ker_width = 3

    # this is to really eliminate the size of image 
    self.pool = nn.MaxPool2d(3)

    self.fc1 = nn.Linear(channels*10*10, 16)
    self.fc2 = nn.Linear(16, 10)

  def forward(self, x):
    x = self.pool(x)
    x = torch.flatten(x, 1)
    x = self.fc1(x)
    x = F.relu(x)
    x = self.fc2(x)
    output = F.log_softmax(x, dim=1)
    return output
# end CloseLayer

class StepLayer(nn.Module):
  def __init__(self,channels):
    super(StepLayer, self).__init__()
    ker_width = 3
    self.conv1 = nn.Conv2d(channels,channels,ker_width,padding=1)
    self.bn1 = nn.BatchNorm2d(channels)
    self.conv2 = nn.Conv2d(channels,channels,ker_width,padding=1)
    self.bn2 = nn.BatchNorm2d(channels)

  def forward(self, x):
    return F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))))
# end StepLayer

class ParallelNet(nn.Module):
  def __init__(self,channels=12,local_steps=8,Tf=1.0,max_levels=1,max_iters=1,print_level=0):
    super(ParallelNet, self).__init__()

    step_layer = lambda: StepLayer(channels)

    self.parallel_nn = torchbraid.LayerParallel(MPI.COMM_WORLD,step_layer,local_steps,Tf,max_fwd_levels=max_levels,max_bwd_levels=max_levels,max_iters=max_iters)
    self.parallel_nn.setPrintLevel(print_level)
    self.parallel_nn.setCFactor(2)
    self.parallel_nn.setSkipDowncycle(True)
    
    self.parallel_nn.setFwdNumRelax(1)         # FCF elsewhere
    self.parallel_nn.setFwdNumRelax(0,level=0) # F-Relaxation on the fine grid for forward solve

    self.parallel_nn.setBwdNumRelax(1)         # FCF elsewhere
    self.parallel_nn.setBwdNumRelax(0,level=0) # F-Relaxation on the fine grid for backward solve

    # this object ensures that only the LayerParallel code runs on ranks!=0
    compose = self.compose = self.parallel_nn.comp_op()
    
    # by passing this through 'compose' (mean composition: e.g. OpenLayer o channels) 
    # on processors not equal to 0, these will be None (there are no parameters to train there)
    self.open_nn = compose(OpenLayer,channels)
    self.close_nn = compose(CloseLayer,channels)
 
  def forward(self, x):
    # by passing this through 'o' (mean composition: e.g. self.open_nn o x) 
    # this makes sure this is run on only processor 0

    x = self.compose(self.open_nn,x)
    x = self.parallel_nn(x)
    x = self.compose(self.close_nn,x)

    return x
# end ParallelNet 
####################################################################################
####################################################################################


####################################################################################
####################################################################################
# Classes and functions that define the basic operations of MG/Opt for TorchBraid.
# Could go into separate file

##
# Linear algebra functions
def tensor_list_dot(v, w):
  ''' Compute dot product of two vectors, v and w, where each vector is a list of tensors '''
  return sum([ torch.dot(vv.flatten(), ww.flatten()) for (vv,ww) in zip(v, w) ])

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
# Nested iteration functions
def train_epoch(rank, model, train_loader, optimizer, epoch, criterion, compose, log_interval):
  ''' Carry out one complete training epoch '''
  model.train()
  
  total_time = 0.0
  for batch_idx, (data, target) in enumerate(train_loader):
    start_time = timer()
    optimizer.zero_grad()
    output = model(data)

    loss = compose(criterion,output,target)
    loss.backward()
    stop_time = timer()
    optimizer.step()

    total_time += stop_time-start_time
    if batch_idx % log_interval == 0:
      root_print(rank,'  Train Epoch: {} [{}/{} ({:.0f}%)]     \tLoss: {:.6f}\tTime Per Batch {:.6f}'.format(
          epoch, batch_idx * len(data), len(train_loader.dataset),
          100. * batch_idx / len(train_loader), loss.item(), total_time/(batch_idx+1.0)))

  root_print(rank,'  Train Epoch: {} [{}/{} ({:.0f}%)]     \tLoss: {:.6f}\tTime Per Batch {:.6f}'.format(
    epoch, (batch_idx+1) * len(data), len(train_loader.dataset),
    100. * (batch_idx+1) / len(train_loader), loss.item(), total_time/(batch_idx+1.0)))


def test(rank, model, test_loader,compose):
  ''' Compute loss and accuracy '''
  model.eval()
  test_loss = 0
  correct = 0
  criterion = nn.CrossEntropyLoss()
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data, target
      output = model(data)
      test_loss += compose(criterion,output,target).item()
       
      output = MPI.COMM_WORLD.bcast(output,root=0)
      pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
      correct += pred.eq(target.view_as(pred)).sum().item()

  test_loss /= len(test_loader.dataset)

  root_print(rank,'  Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
      test_loss, correct, len(test_loader.dataset),
      100. * correct / len(test_loader.dataset)))






##
# Interp / restrict functions
def piecewise_const_interp_network_params(model, cf=2, deep_copy=False, grad=False):
  
  ''' 
  Interpolate the model parameters according to coarsening-factor in time cf.
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
    for child in model.children():
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


def piecewise_const_restrict_network_params(model, cf=2, deep_copy=False, grad=False):
  ''' 
  Restrict the model parameters according to coarsening-factor in time cf.
  Return a list of the restricted model parameters.

  If deep_copy is True, return a deep copy.
  If grad is True, return the network gradient instead
  '''
  
  restrict_params = []

  # loop over all the children, restricting 
  for child in model.children():
    
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


def piecewise_const_restrict_network_state(model_fine, model_coarse, cf=2):
  ''' 
  Restrict the model state according to coarsening-factor in time cf.
  The restricted model state is placed inside of model_coarse
  '''
  
  ##
  # Inject network state to model_coarse.  
  # Note:  We need access to cython cdef to do this, so we put the restrict/interp inside torchbraid_app.pyx
  model_coarse.parallel_nn.fwd_app.inject_network_state(  model_fine.parallel_nn.fwd_app, cf )  
  model_coarse.parallel_nn.bwd_app.inject_network_state(  model_fine.parallel_nn.bwd_app, cf )  


def piecewise_const_interp_network_state(model_fine, model_coarse, cf=2):
  ''' 
  Interp the model state according to coarsening-factor in time cf.
  The interpolated model state is placed inside of model_fine 
  '''
  
  ##
  # interp network state to model_fine.  
  # Note:  We need access to cython cdef to do this, so we put the restrict/interp inside torchbraid_app.pyx
  model_fine.parallel_nn.fwd_app.interp_network_state(  model_coarse.parallel_nn.fwd_app, cf )  
  model_fine.parallel_nn.bwd_app.interp_network_state(  model_coarse.parallel_nn.bwd_app, cf )  


def write_network_params_inplace(model, new_params):
  '''
  Write the parameters of model in-place, overwriting with new_params
  '''
  
  with torch.no_grad():
    old_params = list(model.parameters())
    
    assert(len(old_params) == len(new_params)) 
    
    for (op, np) in zip(old_params, new_params):
      op[:] = np[:]

def get_network_params(model, deep_copy=False, grad=False):
  
  if deep_copy:
    if grad: pp = [torch.clone(params.grad) for params in model.parameters() ]
    else:    pp = [torch.clone(params)      for params in model.parameters() ]
  else:
    if grad: pp = [params.grad for params in model.parameters() ]
    else:    pp = [params      for params in model.parameters() ]
  ##

  return pp




##
# Basic cross Entropy loss function extended to take MG/Opt Term
def basic_mgopt_cross_ent(output, target, network_parameters=None, v=None):
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
  if (network_parameters is not None) and (v is not None): 
    mgopt_term = tensor_list_dot(v, network_parameters)
    return loss - mgopt_term
  else:
    return loss




##
# Compute TB fwd and bwd pass 
def compute_fwd_bwd_pass(lvl, optimizer, model, data, target, criterion, compose, v_h):
  '''
  Compute a backward and forward pass for the model.
  if lvl is 0, no MGOPT term is used
  if lvl > 0, incorporate MGOpt term
  '''
  model.train()

  optimizer.zero_grad()
  output = model(data)
  if lvl == 0:
    loss = compose(criterion, output, target)
  else: # add the MG/Opt Term
    x_h = get_network_params(model, deep_copy=False, grad=False)
    loss = compose(criterion, output, target, x_h, v_h)
  ##
  loss.backward()
  return loss

def line_search(lvl, e_h, optimizer, model, data, target, compose, criterion, fine_loss, alpha, n_line_search, v_h):
  '''
  Simple line-search: Add e_h to fine parameters.  If loss has
  diminished, stop.  Else subtract 1/2 of e_h from fine parameters, and
  continue until loss is reduced.
  '''

  # Add error update to x_h
  # alpha*e_h + x_h --> x_h
  x_h = get_network_params(model, deep_copy=False, grad=False)
  tensor_list_AXPY(alpha, e_h, 1.0, x_h, inplace=True)
  
  # Start Line search 
  for m in range(n_line_search):
    #print("line-search, alpha=", alpha)
    with torch.enable_grad():
      loss = compute_fwd_bwd_pass(lvl, optimizer, model, data, target, criterion, compose, v_h)
    
    new_loss = loss.item()
    if new_loss < 0.999*fine_loss:
      # loss is reduced by at least a fixed amount, end line search
      break
    elif m < (n_line_search-1): 
      # loss is NOT reduced, continue line search
      alpha = alpha/2.0
      tensor_list_AXPY(-alpha, e_h, 1.0, x_h, inplace=True)

  return alpha







##
# Parsing functions
def parse_args():
  """
  Return back an args dictionary based on a standard parsing of the command line inputs
  """

  def compute_levels(num_steps,min_coarse_size,cfactor): 
    from math import log, floor 
    # we want to find $L$ such that ( max_L min_coarse_size*cfactor**L <= num_steps)
    levels =  floor(log(num_steps/min_coarse_size,cfactor))+1 

    if levels<1:
      levels = 1
    return levels
  # end compute_levels

  
  # Command line settings
  parser = argparse.ArgumentParser(description='MG/Opt Solver Parameters')
  parser.add_argument('--seed', type=int, default=1, metavar='S',
                      help='random seed (default: 1)')
  parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                      help='how many batches to wait before logging training status')
  
  # artichtectural settings
  parser.add_argument('--steps', type=int, default=4, metavar='N',
                      help='Number of times steps in the resnet layer (default: 4)')
  parser.add_argument('--channels', type=int, default=4, metavar='N',
                      help='Number of channels in resnet layer (default: 4)')
  
  # algorithmic settings (gradient descent and batching
  parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                      help='input batch size for training (default: 50)')
  parser.add_argument('--epochs', type=int, default=2, metavar='N',
                      help='number of epochs to train (default: 2)')
  parser.add_argument('--samp-ratio', type=float, default=1.0, metavar='N',
                      help='number of samples as a ratio of the total number of samples')
  parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                      help='learning rate (default: 0.01)')
  
  # algorithmic settings (parallel or serial)
  parser.add_argument('--force-lp', action='store_true', default=False,
                      help='Use layer parallel even if there is only 1 MPI rank')
  parser.add_argument('--lp-levels', type=int, default=3, metavar='N',
                      help='Layer parallel levels (default: 3)')
  parser.add_argument('--lp-iters', type=int, default=2, metavar='N',
                      help='Layer parallel iterations (default: 2)')
  parser.add_argument('--lp-print', type=int, default=0, metavar='N',
                      help='Layer parallel print level (default: 0)')
  
  # algorithmic settings (nested iteration)
  parser.add_argument('--ni-levels', type=int, default=3, metavar='N',
                      help='Number of nested iteration levels (default: 3)')
  parser.add_argument('--ni-rfactor', type=int, default=2, metavar='N',
                      help='Refinment factor for nested iteration (default: 2)')
  group = parser.add_mutually_exclusive_group(required=False)
  group.add_argument('--ni-fixed-coarse', dest='ni_fixed_coarse', action='store_true',
                      help='Fix the weights on the coarse levels once trained (default: off)')
  group.add_argument('--ni-no-fixed-coarse', dest='ni_fixed_coarse', action='store_false',
                      help='Fix the weights on the coarse levels once trained (default: off)')
  parser.set_defaults(ni_fixed_coarse=False)
  
  ##
  # Do some parameter checking
  rank  = MPI.COMM_WORLD.Get_rank()
  procs = MPI.COMM_WORLD.Get_size()
  args = parser.parse_args()
  
  if args.lp_levels==-1:
    min_coarse_size = 7
    args.lp_levels = compute_levels(args.steps, min_coarse_size, 4)
  
  if args.steps % procs!=0:
    root_print(rank,'Steps must be an even multiple of the number of processors: %d %d' % (args.steps,procs) )
    sys.exit(0)
  
  ni_levels = args.ni_levels
  ni_rfactor = args.ni_rfactor
  if args.steps % ni_rfactor**(ni_levels-1) != 0:
    root_print(rank,'Steps combined with the coarsest nested iteration level must be an even multiple: %d %d %d' % (args.steps,ni_rfactor,ni_levels-1))
    sys.exit(0)
  
  if args.steps / ni_rfactor**(ni_levels-1) % procs != 0:
    root_print(rank,'Coarsest nested iteration must fit on the number of processors')
    sys.exit(0)

  return args


####################################################################################
####################################################################################




####################################################################################
####################################################################################
# Small Helper Functions (could eventually go in a utils.py)

def root_print(rank,s):
  ''' Parallel print routine '''
  if rank==0:
    print(s)


def unpack_arg(v):
  ''' Helper function for unpacking arguments '''
  if isinstance(v, tuple):
      return v[0], v[1]
  else:
      return v, {}

def print_option(o, indent="  "):
  method,args = unpack_arg(o)
  output = indent + method + '\n'
  if args == {}:
    output += indent + indent + "None\n"
  else:
    for a in args:
      output += indent + indent + a + " : " + str(args[a]) + '\n'
  ##
  return output

####################################################################################
####################################################################################



class mgopt_solver:
    """Stores multigrid hierarchy and implements the multigrid cycle.

    The class constructs the cycling process and points to the methods for
    coarse grid solves.  A call to multilevel_solver.solve() is a typical
    access point.  

    Attributes
    ----------
    levels : level array

        Array of level objects that contain the information needed to coarsen,
        interpolate, relax, and do a coarse-grid solve 

    Methods
    -------
    operator_complexity()
        A measure of the size of the multigrid hierarchy.
    solve()
        Iteratively solves the optimization problem 

    """

    class level:
        """Stores one level of the multigrid hierarchy.

        Attributes
        ----------
        model : PyTorch NN model, tested so far with TorchBraid ParallelNet model
        network : tuple describing the model setup parameters 
        interp_netw_params : tuple describing the option selected for interpolating network parameters 
        optim : tuple describing the option selected for the underlying optimizationg method 
        criterion : tuple describing the option selected for the criterion (objective)
        training_params : tuple describing the training options selected

        """

        def __init__(self):
            """Level construct (empty)."""
            pass


    def __init__(self):
        """Class constructor responsible for creating the objecte 

        Parameters
        ----------
        None
      
        """
        self.levels = []


    def __repr__(self):
        """Print basic statistics about the multigrid hierarchy."""

        (total_op_comp, trainable_op_comp, total_params_per_level, trainable_params_per_level) = self.operator_complexity()


        output = '\nMG/Opt Solver\n'
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
 

    def operator_complexity(self):
        """Operator complexity of this multigrid hierarchy.

        Returns 4-Tuple
          - Number of total parameters on level 0     / Total number of parameters on all levels 
          - Number of trainable parameters on level 0 / Number of trainable parameters on all levels 
          - Array of the total param count on each level
          - Array of the trainable param count on each level
        """
        
        total_params_per_level = []
        trainable_params_per_level = []

        for lvl in self.levels:
          model = lvl.model
          total_params = sum(p.numel() for p in model.parameters())
          trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
          counts = MPI.COMM_WORLD.gather((total_params,trainable_params))
          total_params_per_level.append(counts[0][0])
          trainable_params_per_level.append(counts[0][1])

        total_params_per_level = np.array(total_params_per_level) 
        trainable_params_per_level = np.array(trainable_params_per_level) 
        
        if total_params_per_level.shape[0] > 0:
          total_op_comp =  np.sum(total_params_per_level) / total_params_per_level[0]
          trainable_op_comp =  np.sum(trainable_params_per_level) / trainable_params_per_level[0]
          
          return (total_op_comp, trainable_op_comp, total_params_per_level, trainable_params_per_level)
        
        else:
          return (-1, -1)


    def options_used(self):
        """ Print the options selected to form the hierarchy """
        
        for k, lvl in enumerate(self.levels):
          print("MG/Opt parameters from level " + str(k) )
          print( print_option(lvl.network) )
          print( print_option(lvl.interp_netw_params) )
          print( print_option(lvl.optim) )
          print( print_option(lvl.criterion) )
          print( print_option(lvl.training_params) )
          # ADD OTHER PARAMETERS HERE


    def initialize_with_nested_iteration(self, ni_steps, training_setup=None, networks=None, interp_netw_params=None, optims=None, criterions=None, seed=None):
      """
      Use nested iteration to create a hierarchy of models

      Parameters
      ----------
      ni_steps : array
        array of the number of time_steps at each level of nested iteration 

      training_setup : tuple
        tuple of form (train_loader, test_loader, samp_ratio, batch_size, epochs)
          
        train_loader : PyTorch data loader for training
        test_loader  : PyTorch data loader for testing
        samp_ratio   : float, sampling ratio for training data
        batch_size   : int, batch size for training data
        epochs       : int, number of training epochs
        log_interval : int, how often to output timing and loss data

      
      networks : list
        networks[k] describes the network architecture level k in the nested
        iteration hierarchy, starting from coarse to fine. Each list entry is a
        the form  ('string', param_dict), where string is a supported option,
        and takes the parameters in param_dict.


      interp_netw_params : list
        interp_netw_params[k] describes how to interpolate the network
        parameters at level k in the nested iteration hierarchy, starting from
        coarse to fine.  Each list entry is a the form  ('string', param_dict),
        where string is a supported option, and takes the parameters in
        param_dict.

      optims : list
        optims[k] describes the optimization strategy to use at level k in the
        nested iteration hierarchy, starting from coarse to fine.  Each list
        entry is a the form  ('string', param_dict), where string is a
        supported option, and takes the parameters in param_dict.
     
      criterions : list
        criterions[k] describes the criterion or objective function at level k
        in the nested iteration hierarchy, starting from coarse to fine.  Each
        list entry is a the form  ('string', param_dict), where string is a
        supported option, and takes the parameters in param_dict.

      seed : int
        seed for random number generate (e.g., when initializing weights)

      Returns
      -------
      Initialized hierarchy for MG/Opt 
      No direct output. Changes done internally to multilevel solver object

      """
            
      ##
      # Unpack args
      (train_loader, test_loader, samp_ratio, batch_size, epochs, log_interval) = training_setup
      rank  = MPI.COMM_WORLD.Get_rank()
      procs = MPI.COMM_WORLD.Get_size()

      if( len(ni_steps) != len(networks) ):
        raise ValueError('Length of ni_steps must equal length of networks, i.e., you must have a network architecture defined for each level of nested iteration')
      if( len(ni_steps) != len(interp_netw_params) ):
          raise ValueError('Length of ni_steps must equal length of interp_netw_params, i.e., you must have an interpolation strategy defined for each level of nested iteration')
      if( len(ni_steps) != len(optims) ):
          raise ValueError('Length of ni_steps must equal length of optims, i.e., you must have an optimization strategy defined for each level of nested iteration')
      if( len(ni_steps) != len(criterions) ):
          raise ValueError('Length of ni_steps must equal length of criterions, i.e., you must have a criterion (objective) defined for each level of nested iteration')
      
      ##
      # Seed the generator for the below training 
      if seed is not None:
        torch.manual_seed(torchbraid.utils.seed_from_rank(seed, rank))
      
      ##
      # Check ni_steps that it is a constant r_factor
      nlevels = len(ni_steps)
      ni_rfactor = int(max(ni_steps[1:] / ni_steps[:-1]))
      ni_rfactor_min = int(min(ni_steps[1:] / ni_steps[:-1]))
      if( ni_rfactor != ni_rfactor_min):
        raise ValueError('Nested iteration (ni_steps) should use a single constant refinement factor') 
      
      ##
      # Initialize self.levels with nested iteration
      for k, steps in enumerate(ni_steps):
        
        ##
        # Create new model for this level
        model_string, kwargs = unpack_arg(networks[k])
        if model_string == "ParallelNet":
          model = ParallelNet( **kwargs)
          #
          # Only needed if Braid will create a single time-level.  
          # Could be removed in the multilevel case for memory efficiency
          model.parallel_nn.setBwdStorage(0)
        else:
          # Can insert support for other models 
          raise ValueError('Unsupported model: ' + model_string) 


        # Interpolate weights from coarser model to this one 
        if len(self.levels) > 0: 
          interp_string, kwargs = unpack_arg(interp_netw_params[k])
          
          if interp_string == "piecewise_const_interp_network_params":
            new_params = piecewise_const_interp_network_params(self.levels[-1].model, cf=ni_rfactor, deep_copy=True, grad=False, **kwargs)
            write_network_params_inplace(model, new_params)
            del new_params

          else:
            # Can insert support for other interpolations 
            raise ValueError('Unsupported interpolation: ' + interp_string) 


        ##
        # Diagnostic printing
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        counts = MPI.COMM_WORLD.gather((total_params,trainable_params))
        root_print(rank,'\nNested Iter Level:  ' + str(k) )
        root_print(rank,'  optimizing %d steps' % steps)
        root_print(rank,'  total params: {}'.format([c[0] for c in counts]))
        root_print(rank,'  train params: {}'.format([c[1] for c in counts]))
        root_print(rank,'')


        ##
        # Select optimization method
        optim_string, kwargs = unpack_arg(optims[k])
        if optim_string == "pytorch_sgd":
          optimizer = optim.SGD(model.parameters(), **kwargs)
        else:
          # Can insert support for other optimizers 
          raise ValueError('Unsupported optimizer: ' + optim_string) 
        

        ##
        # Select criterion (objective) and must also select a "compose" function
        criterion_string, kwargs = unpack_arg(criterions[k])
        if criterion_string == "basic_mgopt_cross_ent":
          criterion = basic_mgopt_cross_ent
          compose = model.compose
        else:
          # Can insert support for other criterion (objectives) and compose
          raise ValueError('Unsupported optimizer: ' + optim_string) 
        
        epoch_times = []
        test_times = []
        for epoch in range(1, epochs + 1):
          start_time = timer()
          # train_epoch() is designed to be general for PyTorch networks
          train_epoch(rank, model, train_loader, optimizer, epoch, criterion, compose, log_interval)
          end_time = timer()
          epoch_times.append( end_time-start_time )
      
          # test() is designed to be general for PyTorch networks
          start_time = timer()
          test(rank, model, test_loader, compose)
          end_time = timer()
          test_times.append( end_time-start_time )
        
        ##
        # Store model and parameters
        self.levels.append(self.level())
        self.levels[-1].model = model
        self.levels[-1].network = networks[k]
        self.levels[-1].interp_netw_params = interp_netw_params[k]
        self.levels[-1].optim = optims[k]
        self.levels[-1].criterion = criterions[k]
        self.levels[-1].training_params = ('training params', 
                {'samp_ratio':samp_ratio, 'batch_size':batch_size, 'epochs':epochs, 'log_interval':log_interval})

        # Print epoch-level time averages
        root_print(rank,'  Time per epoch: %.2e (1 std dev %.2e)' % (stats.mean(epoch_times),stats.stdev(epoch_times)))
        root_print(rank,'  Time per test:  %.2e (1 std dev %.2e)' % (stats.mean(test_times), stats.stdev(test_times)))


      ##
      # Reverse order, so finest level is level 0 
      self.levels.reverse()


    def solve(self, b, x0=None, tol=1e-5, maxiter=100, cycle='V', accel=None,
              callback=None, residuals=None, return_residuals=False):
        """Execute multigrid cycling.

        Parameters
        ----------
        b : array
            Right hand side.
        x0 : array
            Initial guess.
        tol : float
            Stopping criteria: relative residual r[k]/r[0] tolerance.
        maxiter : int
            Stopping criteria: maximum number of allowable iterations.
        cycle : {'V','W','F','AMLI'}
            Type of multigrid cycle to perform in each iteration.
        accel : string, function
            Defines acceleration method.  Can be a string such as 'cg'
            or 'gmres' which is the name of an iterative solver in
            pyamg.krylov (preferred) or scipy.sparse.linalg.isolve.
            If accel is not a string, it will be treated like a function
            with the same interface provided by the iterative solvers in SciPy.
        callback : function
            User-defined function called after each iteration.  It is
            called as callback(xk) where xk is the k-th iterate vector.
        residuals : list
            List to contain residual norms at each iteration.

        Returns
        -------
        x : array
            Approximate solution to Ax=b

        See Also
        --------
        aspreconditioner

        Examples
        --------
        >>> from numpy import ones
        >>> from pyamg import ruge_stuben_solver
        >>> from pyamg.gallery import poisson
        >>> A = poisson((100, 100), format='csr')
        >>> b = A * ones(A.shape[0])
        >>> ml = ruge_stuben_solver(A, max_coarse=10)
        >>> residuals = []
        >>> x = ml.solve(b, tol=1e-12, residuals=residuals) # standalone solver

        """

        ######
        # No levelize, just check that it's the right length
        
        # rename solve to optimize ... ?

        from pyamg.util.linalg import residual_norm, norm

        if x0 is None:
            x = np.zeros_like(b)
        else:
            x = np.array(x0)  # copy

        cycle = str(cycle).upper()

        # AMLI cycles require hermitian matrix
        if (cycle == 'AMLI') and hasattr(self.levels[0].A, 'symmetry'):
            if self.levels[0].A.symmetry != 'hermitian':
                raise ValueError('AMLI cycles require \
                    symmetry to be hermitian')

        if accel is not None:

            # Check for symmetric smoothing scheme when using CG
            if (accel is 'cg') and (not self.symmetric_smoothing):
                warn('Incompatible non-symmetric multigrid preconditioner '
                     'detected, due to presmoother/postsmoother combination. '
                     'CG requires SPD preconditioner, not just SPD matrix.')

            # Check for AMLI compatability
            if (accel != 'fgmres') and (cycle == 'AMLI'):
                raise ValueError('AMLI cycles require acceleration (accel) '
                                 'to be fgmres, or no acceleration')

            # py23 compatibility:
            try:
                basestring
            except NameError:
                basestring = str

            # Acceleration is being used
            if isinstance(accel, basestring):
                from pyamg import krylov
                from scipy.sparse.linalg import isolve
                if hasattr(krylov, accel):
                    accel = getattr(krylov, accel)
                else:
                    accel = getattr(isolve, accel)

            A = self.levels[0].A
            M = self.aspreconditioner(cycle=cycle)

            try:  # try PyAMG style interface which has a residuals parameter
                return accel(A, b, x0=x0, tol=tol, maxiter=maxiter, M=M,
                             callback=callback, residuals=residuals)[0]
            except BaseException:
                # try the scipy.sparse.linalg.isolve style interface,
                # which requires a call back function if a residual
                # history is desired

                cb = callback
                if residuals is not None:
                    residuals[:] = [residual_norm(A, x, b)]

                    def callback(x):
                        if sp.isscalar(x):
                            residuals.append(x)
                        else:
                            residuals.append(residual_norm(A, x, b))
                        if cb is not None:
                            cb(x)

                return accel(A, b, x0=x0, tol=tol, maxiter=maxiter, M=M,
                             callback=callback)[0]

        else:
            # Scale tol by normb
            # Don't scale tol earlier. The accel routine should also scale tol
            normb = norm(b)
            if normb != 0:
                tol = tol * normb

        if return_residuals:
            warn('return_residuals is deprecated.  Use residuals instead')
            residuals = []
        if residuals is None:
            residuals = []
        else:
            residuals[:] = []

        # Create uniform types for A, x and b
        # Clearly, this logic doesn't handle the case of real A and complex b
        from scipy.sparse.sputils import upcast
        from pyamg.util.utils import to_type
        tp = upcast(b.dtype, x.dtype, self.levels[0].A.dtype)
        [b, x] = to_type(tp, [b, x])
        b = np.ravel(b)
        x = np.ravel(x)

        A = self.levels[0].A

        residuals.append(residual_norm(A, x, b))

        self.first_pass = True

        while len(residuals) <= maxiter and residuals[-1] > tol:
            if len(self.levels) == 1:
                # hierarchy has only 1 level
                x = self.coarse_solver(A, b)
            else:
                self.__solve(0, x, b, cycle)

            residuals.append(residual_norm(A, x, b))

            self.first_pass = False

            if callback is not None:
                callback(x)

        if return_residuals:
            return x, residuals
        else:
            return x

    def __solve(self, lvl, x, b, cycle):
        """Multigrid cycling.

        Parameters
        ----------
        lvl : int
            Solve problem on level `lvl`
        x : numpy array
            Initial guess `x` and return correction
        b : numpy array
            Right-hand side for Ax=b
        cycle : {'V','W','F','AMLI'}
            Recursively called cycling function.  The
            Defines the cycling used:
            cycle = 'V',    V-cycle
            cycle = 'W',    W-cycle
            cycle = 'F',    F-cycle
            cycle = 'AMLI', AMLI-cycle

        """
        A = self.levels[lvl].A

        self.levels[lvl].presmoother(A, x, b)

        residual = b - A * x

        coarse_b = self.levels[lvl].R * residual
        coarse_x = np.zeros_like(coarse_b)

        if lvl == len(self.levels) - 2:
            coarse_x[:] = self.coarse_solver(self.levels[-1].A, coarse_b)
        else:
            if cycle == 'V':
                self.__solve(lvl + 1, coarse_x, coarse_b, 'V')
            elif cycle == 'W':
                self.__solve(lvl + 1, coarse_x, coarse_b, cycle)
                self.__solve(lvl + 1, coarse_x, coarse_b, cycle)
            elif cycle == 'F':
                self.__solve(lvl + 1, coarse_x, coarse_b, cycle)
                self.__solve(lvl + 1, coarse_x, coarse_b, 'V')
            elif cycle == "AMLI":
                # Run nAMLI AMLI cycles, which compute "optimal" corrections by
                # orthogonalizing the coarse-grid corrections in the A-norm
                nAMLI = 2
                Ac = self.levels[lvl + 1].A
                p = np.zeros((nAMLI, coarse_b.shape[0]), dtype=coarse_b.dtype)
                beta = np.zeros((nAMLI, nAMLI), dtype=coarse_b.dtype)
                for k in range(nAMLI):
                    # New search direction --> M^{-1}*residual
                    p[k, :] = 1
                    self.__solve(lvl + 1, p[k, :].reshape(coarse_b.shape),
                                 coarse_b, cycle)

                    # Orthogonalize new search direction to old directions
                    for j in range(k):  # loops from j = 0...(k-1)
                        beta[k, j] = np.inner(p[j, :].conj(), Ac * p[k, :]) /\
                            np.inner(p[j, :].conj(), Ac * p[j, :])
                        p[k, :] -= beta[k, j] * p[j, :]

                    # Compute step size
                    Ap = Ac * p[k, :]
                    alpha = np.inner(p[k, :].conj(), np.ravel(coarse_b)) /\
                        np.inner(p[k, :].conj(), Ap)

                    # Update solution
                    coarse_x += alpha * p[k, :].reshape(coarse_x.shape)

                    # Update residual
                    coarse_b -= alpha * Ap.reshape(coarse_b.shape)
            else:
                raise TypeError('Unrecognized cycle type (%s)' % cycle)

        x += self.levels[lvl].P * coarse_x   # coarse grid correction

        self.levels[lvl].postsmoother(A, x, b)


