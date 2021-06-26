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


# Clean up excess code in both files, especially main_mgopt2.py, and especially with imports,  
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
# These top functions are PyTorch general, so one file for that, and then the TB specific functions go into that file?

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
# PyTorch train and test network functions (used by nested iteration) 
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
# PyTorch get and write params functions
def write_params_inplace(model, new_params):
  '''
  Write the parameters of model in-place, overwriting with new_params
  '''
  
  with torch.no_grad():
    old_params = list(model.parameters())
    
    assert(len(old_params) == len(new_params)) 
    
    for (op, np) in zip(old_params, new_params):
      op[:] = np[:]

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







##
# TorchBraid Interp / restrict functions
def tb_get_injection_interp_params(model, cf=2, deep_copy=False, grad=False):
  
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


def tb_get_injection_restrict_params(model, cf=2, deep_copy=False, grad=False):
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


##
# Basic TorchBraid cross Entropy loss function extended to take MG/Opt Term
def tb_mgopt_cross_ent(output, target, network_parameters=None, v=None):
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
    x_h = get_params(model, deep_copy=False, grad=False)
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
  x_h = get_params(model, deep_copy=False, grad=False)
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
    interpolate and relax.  See levels definition below.

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
    model : PyTorch NN model, tested so far with TorchBraid ParallelNet model
    network : tuple describing the model setup parameters 
    interp_params : tuple describing the option selected for interpolating network parameters 
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
      print( print_option(lvl.interp_params) )
      print( print_option(lvl.optim) )
      print( print_option(lvl.criterion) )
      print( print_option(lvl.training_params) )
      # ADD OTHER PARAMETERS HERE


  def initialize_with_nested_iteration(self, ni_steps, training_setup=None, networks=None, interp_params=None, optims=None, criterions=None, seed=None):
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
      iteration hierarchy, starting from coarse to fine. 

    interp_params : list
      interp_params[k] describes how to interpolate the network
      parameters at level k in the nested iteration hierarchy, starting from
      coarse to fine.  

    optims : list
      optims[k] describes the optimization strategy to use at level k in the
      nested iteration hierarchy, starting from coarse to fine.  
   
    criterions : list
      criterions[k] describes the criterion or objective function at level k
      in the nested iteration hierarchy, starting from coarse to fine.  
      --> If writing a new criterion, it needs to support two modes.  The
          classic criterion(output, target), and a mode that supports the
          additional MG/Opt term, criterion(output, target, x_h, v_h)

    seed : int
      seed for random number generate (e.g., when initializing weights)
  

    Notes
    -----
    The list entries above are desiged to be of this format, 
      ('string', param_dict), 
    where string is a supported option, and takes parameters 'param_dict'

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
    if( len(ni_steps) != len(interp_params) ):
        raise ValueError('Length of ni_steps must equal length of interp_params, i.e., you must have an interpolation strategy defined for each level of nested iteration')
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
        model.parallel_nn.setBwdStorage(0)  # Only really needed if Braid will create a single time-level.  
      else:
        raise ValueError('Unsupported model: ' + model_string) # can insert new options here

      ##
      # Interpolate weights from coarser model to the new model
      if len(self.levels) > 0: 
        interp_string, kwargs = unpack_arg(interp_params[k])
        if interp_string == "tb_injection_interp_params":
          get_interp_params = tb_get_injection_interp_params
          kwargs.update({'deep_copy' : True, 'grad' : False, 'cf' : ni_rfactor})
        else:
          raise ValueError('Unsupported interpolation: ' + interp_string)  # can insert new options here 
        ##
        new_params = get_interp_params(self.levels[-1].model, **kwargs)
        write_params_inplace(model, new_params)
        del new_params


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
        raise ValueError('Unsupported optimizer: ' + optim_string)  # can insert new options here
      

      ##
      # Select criterion (objective) and must also select a "compose" function
      criterion_string, kwargs = unpack_arg(criterions[k])
      if criterion_string == "tb_mgopt_cross_ent":
        criterion = tb_mgopt_cross_ent
        compose = model.compose
      else:
        raise ValueError('Unsupported optimizer: ' + optim_string)  # can insert new options here
      
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
      self.levels[-1].interp_params = interp_params[k]
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


  def mgopt_solve(self, maxiter=2, epochs=1, mgopt_tol=0, nrelax_pre=1,
          nrelax_post=1, nrelax_coarse=5, mgopt_printlevel=1,
          restrict_params=None, restrict_grads=None, restrict_states=None,
          interp_states=None, line_search=None):

    """
    Use nested iteration to create a hierarchy of models

    Parameters
    ----------
    maxiter : int
      Maximum number of MG/Opt iterations

    epochs : int
      Number of epochs to run on each batch inside of MG/Opt 

    mgopt_tol : float
      If the objective evaluation (loss) drops below this value, then halt 

    nrelax_pre : int
      Number of pre-relaxation steps

    nrelax_post : int
      Number of post-elaxation steps
  
    nrelax_coarse : int
      Number of relaxation steps used to solve the coarse-level 

    mgopt_printlevel : int
      output level for mgopt, if set to 0, turns off output

    restrict_params : list
      restrict_params[k] describes the strategy for restricting network
      parameters on level k in the MG/Opt hierarchy 

    restrict_grads : list
      restrict_grads[k] describes the strategy for restricting network
      gradients on level k in the MG/Opt hierarchy 

    restrict_states : list
      restrict_states[k] describes the strategy for restricting network
      states on level k in the MG/Opt hierarchy 

    interp_states : list
      interpt_states[k] describes the strategy for interpolating network
      states on level k in the MG/Opt hierarchy 

    line_search : list
      line_search[k] describes the strategy for line search with the
      coarse-grid correction on level k in the MG/Opt hierarchy 

    Notes
    -----
    The list entries above are desiged to be of this format, 
      ('string', param_dict), 
    where string is a supported option, and takes parameters 'param_dict'

    Returns
    -------
    Trains the hiearchy in place.  Look in self.levels[i].model for the
    trained model on level i, with i=0 the finest level. 

    """

    # have __solve return a loss.item() and put into losses
    # make sure that the above params are integrated into all steps of mgopt

    # Second, move the MGOPT stuff
    #  Initially two-level
    #  MAKE sure all the "extra" niceties of NI carry on down here, 
    #     like storing params, 
    #
    #  Make sure two codes give the same result
    #


    if( len(self.levels) != len(restrict_params) ):
      raise ValueError('Length of restrict_params must equal number of levels, i.e., you must have a restriction for parameters defined for each level of nested iteration')
    if( len(self.levels) != len(restrict_grads) ):
      raise ValueError('Length of restrict_grads must equal number of levels, i.e., you must have a restriction for gradients defined for each level of nested iteration')
    if( len(self.levels) != len(restrict_states) ):
      raise ValueError('Length of restrict_states must equal number of levels, i.e., you must have a restriction for states defined for each level of nested iteration')
    if( len(self.levels) != len(interp_states) ):
      raise ValueError('Length of interp_states must equal number of levels, i.e., you must have an interpolation for states defined for each level of nested iteration')
    if( len(self.levels) != len(line_search) ):
      raise ValueError('Length of line_search must equal number of levels, i.e., you must have a line search defined for each level of nested iteration')
    

  losses = []

    while len(losses) <= maxiter:
      ll = self.__solve(0,  ... 
              #===> Insert recursive params like lvl and vh
      losses.append(ll)

      if losses[-1] < mgopt_tol:
        break

      # This print needs to happen in the solve ... not __solve, should only print average time per MG/Opt Epoch, and Loss
      # ==> Insert timer statements here to track this, and update the outputs here
      root_print(rank,'Train Epoch: {} [{}/{} ({:.0f}%)]     \tLoss: {:.6f}\tTime Per Batch {:.6f}'.format(
        epoch, (batch_idx+1) * len(data), len(train_loader.dataset),
        100. * (batch_idx+1) / len(train_loader), loss.item(),total_time/(batch_idx+1.0)))


    return losses



  def __solve(self, INSERT recursive parameters   
    lvl = 0    # Recursive parameters to pass in to __solve
    v_h = None

    """
    insert comments 

    """
    
    for epoch in range(1, epochs + 1):   # this loop should actually be the .solve() level ,  epochs is a parameter 
      
      total_time = 0.0
      for batch_idx, (data, target) in enumerate(train_loader):
        
        start = timer()
        for it in range(maxiter):
          if (mgopt_printlevel == 1) and (lvl == 0):  root_print(rank, "\nMG/Opt Iter:  " + str(it) + "   batch_idx:  " + str(batch_idx) ) 
          if mgopt_printlevel == 1:  root_print(rank, "\n  Level:  " + str(lvl)) 
          
          optimizer_fine = optim.SGD(models[lvl].parameters(), lr=lr, momentum=momentum)  # lr and momentum present in levels[i].optim args
          
          #import pdb; pdb.set_trace()
          # 1. relax (carry out optimization steps)
          for k in range(nrelax_pre):
            loss = compute_fwd_bwd_pass(lvl, optimizer_fine, models[lvl], data, target, my_criterion, compose, v_h)
            if (mgopt_printlevel == 1):  root_print(rank, "  Pre-relax loss:       " + str(loss.item()) ) 
            optimizer_fine.step()
       
          # 2. compute new gradient g_h
          # First evaluate network, forward and backward.  Return value is scalar value of the loss.
          loss = compute_fwd_bwd_pass(lvl, optimizer_fine, models[lvl], data, target, my_criterion, compose, v_h)
          fine_loss = loss.item()
          if mgopt_printlevel == 1:  root_print(rank, "  Pre-relax done loss:  " + str(loss.item())) 
          # Second, note that the gradient is waiting in models[lvl], to accessed next
     
          # 3. Restrict 
          #    (i)   Network state (primal and adjoint), 
          #    (ii)  Parameters (x_h), and 
          #    (iii) Gradient (g_h) to H
          # x_H^{zero} = R(x_h)   and    \tilde{g}_H = R(g_h)
          with torch.no_grad():
            tb_injection_restrict_network_state(models[lvl], models[lvl+1], cf=ni_rfactor)
            gtilde_H = tb_get_injection_restrict_params(models[lvl], cf=ni_rfactor, deep_copy=True, grad=True)    # Add cf, deep_copy to kwargs (grad already there) 
            x_H      = tb_get_injection_restrict_params(models[lvl], cf=ni_rfactor, deep_copy=True, grad=False)   # Add cf, deep_copy to kwargs (grad already there) 
            # For x_H to take effect, these parameters must be written to the next coarser network
            write_params_inplace(models[lvl+1], x_H)
            # Must store x_H for later error computation
            x_H_zero = tensor_list_deep_copy(x_H)
      
  
          # 4. compute gradient on coarse level, using restricted parameters
          #  g_H = grad( f_H(x_H) )
          optimizer_coarse = optim.SGD(models[lvl+1].parameters(), lr=lr, momentum=momentum)    # lr and momentum present in levels[i].optim args
          # Evaluate gradient.  For computing fwd_bwd_pass, give 0 as first
          # parameter, so that the MGOpt term is turned-off.  We just want hte
          # gradient of f_H here.
          loss = compute_fwd_bwd_pass(0, optimizer_coarse, models[lvl+1], data, target, my_criterion, compose, None)
          with torch.no_grad():
            g_H = get_params(models[lvl+1], deep_copy=True, grad=True)
      
          # 5. compute coupling term
          #  v = g_H - \tilde{g}_H
          with torch.no_grad():
            v_H = tensor_list_AXPY(1.0, g_H, -1.0, gtilde_H)
      
          # 6. solve coarse-grid (eventually do recursive call if not coarsest level)
          #  x_H = min f_H(x_H) - <v_H, x_H>
          for m in range(nrelax_coarse):
            loss = compute_fwd_bwd_pass(lvl+1, optimizer_coarse, models[lvl+1], data, target, my_criterion, compose, v_H)
            optimizer_coarse.step()
      
          # 7. Interpolate 
          #    (i)  error correction to fine-level, and 
          #    (ii) network state to fine-level (primal and adjoint)
          #  e_h = P( x_H - x_H^{init})
          with torch.no_grad():
            x_H = get_params(models[lvl+1], deep_copy=False, grad=False)
            e_H = tensor_list_AXPY(1.0, x_H, -1.0, x_H_zero)
            #
            # to correctly interpolate e_H --> e_h, we need to put these values in a
            # network, so the interpolation knows the layer-parallel structure and
            # where to refine.
            write_params_inplace(models[lvl+1], e_H)
            e_h = tb_get_injection_interp_params(models[lvl+1], cf=ni_rfactor, deep_copy=True, grad=False)
            tb_injection_interp_network_state(models[lvl], models[lvl+1], cf=ni_rfactor)
            
  
          # 8. apply linesearch to update x_h
          #  x_h = x_h + alpha*e_h
          with torch.no_grad():
            alpha = line_search(lvl, e_h, optimizer_fine, models[lvl], data, target, compose, my_criterion, fine_loss, alpha, n_line_search, v_h )
            root_print(rank, "  LS Alpha used:        " + str(alpha)) 
            alpha = alpha*2
  
          # 9. post-relaxation
          for k in range(nrelax_post):
            loss = compute_fwd_bwd_pass(lvl, optimizer_fine, models[lvl], data, target, my_criterion, compose, v_h)
            if (mgopt_printlevel == 1) and (k==0) :  root_print(rank, "  CG Corr done loss:    " + str(loss.item()) ) 
            elif (mgopt_printlevel == 1):  root_print(rank, "  Post-relax loss:      " + str(loss.item()) )
            optimizer_fine.step()
          
          if (mgopt_printlevel == 1):  
            loss = compute_fwd_bwd_pass(lvl, optimizer_fine, models[lvl], data, target, my_criterion, compose, v_h)
            root_print(rank, "  Post-relax loss:      " + str(loss.item()) )
  
        ## End cycle
        # record timer
        end = timer()
        total_time += (end-start)
        
        if batch_idx % log_interval == 0:  # log_interval is  levels[i].training_params
          if (mgopt_printlevel == 1):  root_print(rank, "\n------------------------------------------------------------------------")
          root_print(rank,'Train Epoch: {} [{}/{} ({:.0f}%)]     \tLoss: {:.6f}\tTime Per Batch {:.6f}'.format(
              epoch, batch_idx * len(data), len(train_loader.dataset),
              100. * batch_idx / len(train_loader), loss.item(), total_time/(batch_idx+1.0)))
          if (mgopt_printlevel == 1):  root_print(rank, "------------------------------------------------------------------------")
      
      ## End Batch loop
       
  
