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

# Future Work:
#
#  - Put together a fixed point test, where a solver is trained on a batch
#    which it correctly classifies.  The coarse-grid correction should be zero. 
#
#  - Could someone look over the line-search, test it out?
#
#  - Any multilevel debug? 
#
#  - Parallel
#
#  - Break this file up into parts?
#
#  - How to structure mini-batch loops, e.g., one loop outside of MG/Opt and/or one loop inside MG/Opt (used by each relaxation step)
#
#  - The data and target could be "coarsened" or sampled differently on
#    each level.  There are not hooks for this right now, but they would be
#    easy to add.
#
#  - Coarsegrid convexity tweak from Nash's original paper 



__all__ = [ 'mgopt_solver', 'parse_args' ]

####################################################################################
####################################################################################
# Classes and functions that define the basic network types for MG/Opt and TorchBraid.

class OpenLayer(nn.Module):
  ''' Opening layer (not ODE-net, not parallelized in time) '''
  def __init__(self,channels):
    super(OpenLayer, self).__init__()
    ker_width = 3
    self.conv = nn.Conv2d(3,channels,ker_width,padding=1)

  def forward(self, x):
    return F.relu(self.conv(x))
# end OpenLayer

class CloseLayer(nn.Module):
  ''' Closing layer (not ODE-net, not parallelized in time) '''
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
  ''' Single ODE-net layer will be parallelized in time ''' 
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
  ''' Full parallel ODE-net based on StepLayer,  will be parallelized in time ''' 
  def __init__(self, channels=12, local_steps=8, Tf=1.0, max_fwd_levels=1, max_bwd_levels=1, max_iters=1, max_fwd_iters=0, 
                     print_level=0, braid_print_level=0, fwd_cfactor=4, bwd_cfactor=4, fine_fwd_fcf=False, 
                     fine_bwd_fcf=False, fwd_nrelax=1, bwd_nrelax=1, skip_downcycle=True, fmg=False, fwd_relax_only_cg=0, 
                     bwd_relax_only_cg=0, CWt=1.0, fwd_finalrelax=False):
    super(ParallelNet, self).__init__()

    step_layer = lambda: StepLayer(channels)

    self.parallel_nn = torchbraid.LayerParallel(MPI.COMM_WORLD,step_layer,local_steps,Tf,max_fwd_levels=max_fwd_levels,max_bwd_levels=max_bwd_levels,max_iters=max_iters)
    if max_fwd_iters>0:
      self.parallel_nn.setFwdMaxIters(max_fwd_iters)
    self.parallel_nn.setPrintLevel(print_level,True)
    #self.parallel_nn.setPrintLevel(2,False)
    self.parallel_nn.setPrintLevel(braid_print_level,False)
    self.parallel_nn.setFwdCFactor(fwd_cfactor)
    self.parallel_nn.setBwdCFactor(bwd_cfactor)
    self.parallel_nn.setSkipDowncycle(skip_downcycle)
    self.parallel_nn.setBwdRelaxOnlyCG(bwd_relax_only_cg)
    self.parallel_nn.setFwdRelaxOnlyCG(fwd_relax_only_cg)
    self.parallel_nn.setCRelaxWt(CWt)
    self.parallel_nn.setMinCoarse(2)

    if fwd_finalrelax:
        self.parallel_nn.setFwdFinalFCRelax()

    if fmg:
      self.parallel_nn.setFMG()

    self.parallel_nn.setFwdNumRelax(fwd_nrelax)  # Set relaxation iters for forward solve 
    self.parallel_nn.setBwdNumRelax(bwd_nrelax)  # Set relaxation iters for backward solve
    if not fine_fwd_fcf:
      self.parallel_nn.setFwdNumRelax(0,level=0) # F-Relaxation on the fine grid for forward solve
    else:
      self.parallel_nn.setFwdNumRelax(1,level=0) # FCF-Relaxation on the fine grid for forward solve
    if not fine_bwd_fcf:
      self.parallel_nn.setBwdNumRelax(0,level=0) # F-Relaxation on the fine grid for backward solve
    else:
      self.parallel_nn.setBwdNumRelax(1,level=0) # FCF-Relaxation on the fine grid for backward solve

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
# Functions to facilitate MG/Opt and PyTorch 

##
# Basic Linear algebra functions
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
# PyTorch train and test network functions
def train_epoch(rank, model, train_loader, optimizer, epoch, criterion, criterion_kwargs, compose, log_interval, mgopt_printlevel):
  ''' Carry out one complete training epoch '''
  model.train()
  
  total_time = 0.0
  for batch_idx, (data, target) in enumerate(train_loader):
    start_time = timer()
    optimizer.zero_grad()
    output = model(data)

    loss = compose(criterion, output, target, **criterion_kwargs)
    loss.backward()
    stop_time = timer()
    optimizer.step()

    total_time += stop_time-start_time
    if batch_idx % log_interval == 0:
      root_print(rank, mgopt_printlevel, 1, '  Train Epoch: {} [{}/{} ({:.0f}%)]     \tLoss: {:.6f}\tTime Per Batch {:.6f}'.format(
          epoch, batch_idx * len(data), len(train_loader.dataset),
          100. * batch_idx / len(train_loader), loss.item(), total_time/(batch_idx+1.0)))
  ##

  root_print(rank, mgopt_printlevel, 1, '  Train Epoch: {} [{}/{} ({:.0f}%)]     \tLoss: {:.6f}\tTime Per Batch {:.6f}'.format(
    epoch, (batch_idx+1) * len(data), len(train_loader.dataset),
    100. * (batch_idx+1) / len(train_loader), loss.item(), total_time/(batch_idx+1.0)))

def test(rank, model, test_loader, criterion, criterion_kwargs, compose, mgopt_printlevel, indent=''):
  ''' Compute loss and accuracy '''
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data, target
      output = model(data)
      test_loss += compose(criterion, output, target, **criterion_kwargs).item()
       
      output = MPI.COMM_WORLD.bcast(output,root=0)
      pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
      correct += pred.eq(target.view_as(pred)).sum().item()

  test_loss /= len(test_loader.dataset)

  root_print(rank, mgopt_printlevel, 1, indent + 'Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
      test_loss, correct, len(test_loader.dataset),
      100. * correct / len(test_loader.dataset)))

def compute_fwd_bwd_pass(lvl, optimizer, model, data, target, criterion, criterion_kwargs, compose, v_h):
  '''
  Compute a backward and forward pass for the model.
  if lvl is 0, no MGOPT term is used
  if lvl > 0, incorporate MGOpt term
  '''
  model.train()

  optimizer.zero_grad()
  output = model(data)
  if lvl == 0:
    loss = compose(criterion, output, target, **criterion_kwargs)
  else: # add the MG/Opt Term
    x_h = get_params(model, deep_copy=False, grad=False)
    loss = compose(criterion, output, target, x_h, v_h, **criterion_kwargs)
  ##
  loss.backward()
  return loss


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

####################################################################################
####################################################################################



####################################################################################
####################################################################################
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


def tb_simple_backtrack_ls(lvl, e_h, x_h, v_h, model, optimizer, data, target, criterion, criterion_kwargs, compose, old_loss, e_dot_gradf, mgopt_printlevel, ls_params):
  '''
  Simple line-search: Add e_h to fine parameters.  If loss has
  diminished, stop.  Else subtract 1/2 of e_h from fine parameters, and
  continue until loss is reduced.
  '''
  rank  = MPI.COMM_WORLD.Get_rank()
  try:
    n_line_search = ls_params['n_line_search']
    alpha = ls_params['alpha']
    c1 = ls_params['c1']
  except:
    raise ValueError('tb_simple_backtrack_ls requires a ls_params dictionary with n_line_search, alpha, and c_1 (Armijo condition) as dictionary keys.')

  # Add error update to x_h
  # alpha*e_h + x_h --> x_h
  tensor_list_AXPY(alpha, e_h, 1.0, x_h, inplace=True)
  
  # Start Line search 
  for m in range(n_line_search):
    #print("line-search, alpha=", alpha)
    with torch.enable_grad():
      loss = compute_fwd_bwd_pass(lvl, optimizer, model, data, target, criterion, criterion_kwargs, compose, v_h)
    
    new_loss = loss.item()

    # Check Wolfe condition (i), also called the Armijo rule
    #  f(x + alpha p) <=  f(x) + c alpha <p, grad f(x) >
    if new_loss < 0.999*old_loss + c1*alpha*e_dot_gradf:
      break
    elif m < (n_line_search-1): 
      # loss is NOT reduced enough, continue line search
      alpha = alpha/2.0
      tensor_list_AXPY(-alpha, e_h, 1.0, x_h, inplace=True)
  ##
  # end for-loop

  # Double alpha, and store for next time in ls_params, before returning
  root_print(rank, mgopt_printlevel, 2, "  LS Alpha used:        " + str(alpha) + "  e_dot_gradf:  " + str(e_dot_gradf) + "  old_loss + c1*alpha*e_dot_gradf = " + str(old_loss + c1*alpha*e_dot_gradf))
  ls_params['alpha'] = alpha*2

####################################################################################
####################################################################################
            




####################################################################################
####################################################################################
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
  parser.add_argument('--tf',type=float,default=1.0,
                      help='Final time')

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
  parser.add_argument('--lp-fwd-levels', type=int, default=3, metavar='N',
                      help='Layer parallel levels for forward solve (default: 3)')
  parser.add_argument('--lp-bwd-levels', type=int, default=3, metavar='N',
                      help='Layer parallel levels for backward solve (default: 3)')
  parser.add_argument('--lp-iters', type=int, default=2, metavar='N',
                      help='Layer parallel iterations (default: 2)')
  parser.add_argument('--lp-fwd-iters', type=int, default=-1, metavar='N',
                      help='Layer parallel (forward) iterations (default: -1, default --lp-iters)')
  parser.add_argument('--lp-print', type=int, default=0, metavar='N',
                      help='Layer parallel internal print level (default: 0)')
  parser.add_argument('--lp-braid-print', type=int, default=0, metavar='N',
                      help='Layer parallel braid print level (default: 0)')
  parser.add_argument('--lp-fwd-cfactor', type=int, default=4, metavar='N',
                      help='Layer parallel coarsening factor for forward solve (default: 4)')
  parser.add_argument('--lp-bwd-cfactor', type=int, default=4, metavar='N',
                      help='Layer parallel coarsening factor for backward solve (default: 4)')
  parser.add_argument('--lp-fwd-nrelax-coarse', type=int, default=1, metavar='N',
                      help='Layer parallel relaxation steps on coarse grids for forward solve (default: 1)')
  parser.add_argument('--lp-bwd-nrelax-coarse', type=int, default=1, metavar='N',
                      help='Layer parallel relaxation steps on coarse grids for backward solve (default: 1)')
  parser.add_argument('--lp-fwd-finefcf',action='store_true', default=False, 
                      help='Layer parallel fine FCF for forward solve, on or off (default: False)')
  parser.add_argument('--lp-bwd-finefcf',action='store_true', default=False, 
                      help='Layer parallel fine FCF for backward solve, on or off (default: False)')
  parser.add_argument('--lp-fwd-finalrelax',action='store_true', default=False, 
                      help='Layer parallel do final FC relax after forward cycle ends (always on for backward). (default: False)')
  parser.add_argument('--lp-use-downcycle',action='store_true', default=False, 
                      help='Layer parallel use downcycle on or off (default: False)')
  parser.add_argument('--lp-use-fmg',action='store_true', default=False, 
                      help='Layer parallel use FMG for one cycle (default: False)')
  parser.add_argument('--lp-bwd-relaxonlycg',action='store_true', default=0, 
                      help='Layer parallel use relaxation only on coarse grid for backward cycle (default: False)')
  parser.add_argument('--lp-fwd-relaxonlycg',action='store_true', default=0, 
                      help='Layer parallel use relaxation only on coarse grid for forward cycle (default: False)')
  parser.add_argument('--lp-use-crelax-wt', type=float, default=1.0, metavar='CWt',
                      help='Layer parallel use weighted C-relaxation on backwards solve (default: 1.0).  Not used for coarsest braid level.')

  # algorithmic settings (nested iteration)
  parser.add_argument('--ni-levels', type=int, default=3, metavar='N',
                      help='Number of nested iteration levels (default: 3)')
  parser.add_argument('--ni-rfactor', type=int, default=2, metavar='N',
                      help='Refinment factor for nested iteration (default: 2)')
  
  # algorithmic settings (MG/Opt)
  parser.add_argument('--mgopt-printlevel', type=int, default=1, metavar='N',
                      help='Print level for MG/Opt, 0 least, 1 some, 2 a lot') 
  parser.add_argument('--mgopt-iter', type=int, default=1, metavar='N',
                      help='Number of MG/Opt iterations to optimize over a batch')
  parser.add_argument('--mgopt-levels', type=int, default=2, metavar='N',
                      help='Number of MG/Opt levels to use')
  parser.add_argument('--mgopt-nrelax-pre', type=int, default=1, metavar='N',
                      help='Number of MG/Opt pre-relaxations on each level')
  parser.add_argument('--mgopt-nrelax-post', type=int, default=1, metavar='N',
                      help='Number of MG/Opt post-relaxations on each level')
  parser.add_argument('--mgopt-nrelax-coarse', type=int, default=3, metavar='N',
                      help='Number of MG/Opt relaxations to solve the coarsest grid')


  ##
  # Do some parameter checking
  rank  = MPI.COMM_WORLD.Get_rank()
  procs = MPI.COMM_WORLD.Get_size()
  args = parser.parse_args()
  
  if args.lp_bwd_levels==-1:
    min_coarse_size = 3
    args.lp_bwd_levels = compute_levels(args.steps, min_coarse_size, args.lp_bwd_cfactor)

  if args.lp_fwd_levels==-1:
    min_coarse_size = 3
    args.lp_fwd_levels = compute_levels(args.steps,min_coarse_size,args.lp_fwd_cfactor)

  if args.steps % procs!=0:
    root_print(rank, 1, 1, 'Steps must be an even multiple of the number of processors: %d %d' % (args.steps,procs) )
    sys.exit(0)
  
  ni_levels = args.ni_levels
  ni_rfactor = args.ni_rfactor
  if args.steps % ni_rfactor**(ni_levels-1) != 0:
    root_print(rank, 1, 1, 'Steps combined with the coarsest nested iteration level must be an even multiple: %d %d %d' % (args.steps,ni_rfactor,ni_levels-1))
    sys.exit(0)
  
  if args.steps / ni_rfactor**(ni_levels-1) % procs != 0:
    root_print(rank, 1, 1, 'Coarsest nested iteration must fit on the number of processors')
    sys.exit(0)

  return args


####################################################################################
####################################################################################




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
      print(s)

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
    model           : PyTorch NN model, tested so far with TorchBraid ParallelNet model defined above
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
    """

    def __init__(self):
      """Level construct (empty)."""
      pass


  def __init__(self):
    """ MG/Oopt constructor """
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
    
    def print_option(o, indent="  ", attr_name=""):
      ''' Helper function to print the user-defined options to a formatted string'''
      method,args = unpack_arg(o)
      output = indent + attr_name + method + '\n' 
      if args == {}:
        output += indent + indent + "Parameters: None\n"
      else:
        for a in args:
          output += indent + indent +a + " : " + str(args[a]) + '\n'
      ##
      return output


    rank  = MPI.COMM_WORLD.Get_rank()
    output = ""
    for k, lvl in enumerate(self.levels):
      output = output + "MG/Opt parameters from level " + str(k)
      if hasattr(self.levels[k], 'network'): output = output + print_option(lvl.network, attr_name="network: ") + '\n' 
      if hasattr(self.levels[k], 'interp_params'): output = output + print_option(lvl.interp_params, attr_name="interp_params: ") + '\n'
      if hasattr(self.levels[k], 'optim'): output = output + print_option(lvl.optims, attr_name="optim: ") + '\n'
      if hasattr(self.levels[k], 'criterion'): output = output + print_option(lvl.criterions, attr_name="criterion: ") + '\n'
      if hasattr(self.levels[k], 'restrict_params'): output = output + print_option(lvl.restrict_params, attr_name="restrict_params: ") + '\n'
      if hasattr(self.levels[k], 'restrict_grads'): output = output + print_option(lvl.restrict_grads, attr_name="restrict_grads: ") + '\n'
      if hasattr(self.levels[k], 'restrict_states'): output = output + print_option(lvl.restrict_states, attr_name="restrict_states: ") + '\n'
      if hasattr(self.levels[k], 'interp_states'): output = output + print_option(lvl.interp_states, attr_name="interp_states: ") + '\n'
      if hasattr(self.levels[k], 'line_search'): output = output + print_option(lvl.line_search, attr_name="line_search: ") + '\n'
    ##
    root_print(rank, 1, 1, output)
      


  def initialize_with_nested_iteration(self, ni_steps, 
                                       train_loader, 
                                       test_loader, 
                                       networks, 
                                       epochs           = 1, 
                                       log_interval     = 1, 
                                       mgopt_printlevel = 1,
                                       interp_params    = "tb_get_injection_interp_params", 
                                       optims           = ("pytorch_sgd", { 'lr':0.01, 'momentum':0.9}), 
                                       criterions       = "tb_mgopt_cross_ent", 
                                       seed             = None):
    """
    Use nested iteration to create a hierarchy of models

    Parameters
    ----------
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

    seed : int
      seed for random number generate (e.g., when initializing weights)
  

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
    rank  = MPI.COMM_WORLD.Get_rank()
    procs = MPI.COMM_WORLD.Get_size()

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
    ni_rfactor = int(max(ni_steps[1:] / ni_steps[:-1]))
    ni_rfactor_min = int(min(ni_steps[1:] / ni_steps[:-1]))
    self.ni_steps = ni_steps
    self.ni_rfactor = ni_rfactor
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
        raise ValueError('Unsupported model: ' + model_string)

      ##
      # Select Interpolate weights from coarser model to the new model
      if len(self.levels) > 0: 
        (get_interp_params, interp_params_kwargs) = self.process_get_interp_params(interp_params[k])
        new_params = get_interp_params(self.levels[-1].model, **interp_params_kwargs)
        write_params_inplace(model, new_params)
        del new_params

      ##
      # Diagnostic printing
      total_params = sum(p.numel() for p in model.parameters())
      trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
      counts = MPI.COMM_WORLD.gather((total_params,trainable_params))
      root_print(rank, mgopt_printlevel, 1, '\nNested Iter Level:  ' + str(k) )
      root_print(rank, mgopt_printlevel, 1, '  optimizing %d steps' % steps)
      root_print(rank, mgopt_printlevel, 1, '  total params: {}'.format([c[0] for c in counts]))
      root_print(rank, mgopt_printlevel, 1, '  train params: {}'.format([c[1] for c in counts]))
      root_print(rank, mgopt_printlevel, 1, '')

      ##
      # Select Optimization method
      (optimizer, optim_kwargs) = self.process_optimizer(optims[k], model)

      ##
      # Select Criterion (objective) and compose function
      (criterion, compose, criterion_kwargs) = self.process_criterion(criterions[k], model)
      
      ##
      # Begin epoch loop
      epoch_times = []
      test_times = []
      for epoch in range(1, epochs + 1):
        start_time = timer()
        # train_epoch() is designed to be general for PyTorch networks
        train_epoch(rank, model, train_loader, optimizer, epoch, criterion, criterion_kwargs, compose, log_interval, mgopt_printlevel)
        end_time = timer()
        epoch_times.append( end_time-start_time )
    
        # test() is designed to be general for PyTorch networks
        start_time = timer()
        test(rank, model, test_loader, criterion, criterion_kwargs, compose, mgopt_printlevel, indent='\n  ')
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

      # Print epoch-level time averages
      if epochs == 1:
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
                        restrict_params = ("tb_get_injection_restrict_params", {'grad' : False}), 
                        restrict_grads = ("tb_get_injection_restrict_params", {'grad' : True}), 
                        restrict_states = "tb_injection_restrict_network_state",
                        interp_states = "tb_injection_interp_network_state", 
                        line_search = ("tb_simple_backtrack_ls", {'ls_params' : {'n_line_search' : 6, 'alpha' : 1.0, 'c1' : 0.0}} ) 
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
      restricted_params   = get_restrict_params(model, **kwargs)
      restricted_grad     = get_restrict_grad(model, **kwargs)
      interpolated_params = get_interp_params(model, **kwargs)

    
    Returns
    -------
    Trains the hiearchy in place.  Look in self.levels[i].model for the
    trained model on level i, with i=0 the finest level. 
    
    List of loss values from each MG/Opt epoch

    """
    
    rank  = MPI.COMM_WORLD.Get_rank()
     
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
    # Set the optimizer on each level (some optimizers preserve state between runs, so we instantiate here)
    for k in range(mgopt_levels):
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
          loss_item = self.__solve(lvl, data, target, v_h, nrelax_pre,
                                   nrelax_post, nrelax_coarse, mgopt_printlevel, 
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
          root_print(rank, mgopt_printlevel, 2, "\n------------------------------------------------------------------------------")
          root_print(rank, mgopt_printlevel, 1, 'Train Epoch: {} [{}/{} ({:.0f}%)]     \tLoss: {:.6f}\tTime Per Batch {:.6f}'.format(
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
      test(rank, self.levels[0].model, test_loader, criterion, criterion_kwargs, compose, mgopt_printlevel, indent='\n  ')
      end = timer()
      test_times.append( end -start )
      
      ##
      # For printlevel 2, also test accuracy ong coarse-levels
      if(mgopt_printlevel >= 2):
        for i, level in enumerate(self.levels[1:]):
          root_print(rank, mgopt_printlevel, 2, '  Test accuracy information for level ' + str(i+1))
          test(rank, level.model, test_loader, criterion, criterion_kwargs, compose, mgopt_printlevel, indent='    ')
      
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


  def __solve(self, lvl, data, target, v_h, nrelax_pre, nrelax_post,
          nrelax_coarse, mgopt_printlevel, mgopt_levels, restrict_params,
          restrict_grads, restrict_states, interp_states, line_search):

    """
    Recursive solve function for carrying out one MG/Opt iteration
    See solve() for parameter description
    """
    
    rank  = MPI.COMM_WORLD.Get_rank()
    model = self.levels[lvl].model
    optimizer = self.levels[lvl].optimizer
    coarse_model = self.levels[lvl+1].model
    coarse_optimizer = self.levels[lvl+1].optimizer
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
    (coarse_criterion, coarse_compose, coarse_criterion_kwargs) = self.process_criterion(self.levels[lvl+1].criterions, model)

    # 1. relax (carry out optimization steps)
    for k in range(nrelax_pre):
      loss = compute_fwd_bwd_pass(lvl, optimizer, model, data, target, criterion, criterion_kwargs, compose, v_h)
      root_print(rank, mgopt_printlevel, 2, "  Pre-relax loss:       " + str(loss.item()) ) 
      optimizer.step()
    
    # 2. compute new gradient g_h
    # First evaluate network, forward and backward.  Return value is scalar value of the loss.
    loss = compute_fwd_bwd_pass(lvl, optimizer, model, data, target, criterion, criterion_kwargs, compose, v_h)
    fine_loss = loss.item()
    root_print(rank, mgopt_printlevel, 2, "  Pre-relax done loss:  " + str(loss.item())) 
    with torch.no_grad():
      g_h = get_params(model, deep_copy=True, grad=True)

    # 3. Restrict 
    #    (i)   Network state (primal and adjoint), 
    #    (ii)  Parameters (x_h), and 
    #    (iii) Gradient (g_h) to H
    # x_H^{zero} = R(x_h)   and    \tilde{g}_H = R(g_h)
    with torch.no_grad():
      do_restrict_states(model, coarse_model, **restrict_states_kwargs)
      gtilde_H = get_restrict_grad(model, **restrict_grad_kwargs)
      x_H      = get_restrict_params(model, **restrict_params_kwargs) 
      # For x_H to take effect, these parameters must be written to the next coarser network
      write_params_inplace(coarse_model, x_H)
      # Must store x_H for later error computation
      x_H_zero = tensor_list_deep_copy(x_H)
    
    # 4. compute gradient on coarse level, using restricted parameters
    #  g_H = grad( f_H(x_H) )
    #
    # Evaluate gradient.  For computing fwd_bwd_pass, give 0 as first
    # parameter, so that the MG/Opt term is turned-off.  We just want hte
    # gradient of f_H here.
    loss = compute_fwd_bwd_pass(0, coarse_optimizer, coarse_model, data, target, coarse_criterion, coarse_criterion_kwargs, coarse_compose, None)
    with torch.no_grad():
      g_H = get_params(coarse_model, deep_copy=True, grad=True)
    
    # 5. compute coupling term
    #  v = g_H - \tilde{g}_H
    with torch.no_grad():
      v_H = tensor_list_AXPY(1.0, g_H, -1.0, gtilde_H)
    
    # 6. solve coarse-grid 
    #  x_H = min f_H(x_H) - <v_H, x_H>
    if (lvl+2) == mgopt_levels:
      # If on coarsest level, do a "solve" by carrying out a number of relaxation steps
      root_print(rank, mgopt_printlevel, 2, "\n  Level:  " + str(lvl+1))
      for m in range(nrelax_coarse):
        loss = compute_fwd_bwd_pass(lvl+1, coarse_optimizer, coarse_model, data, target, coarse_criterion, coarse_criterion_kwargs, coarse_compose, v_H)
        coarse_optimizer.step()
        root_print(rank, mgopt_printlevel, 2, "  Coarsest grid solve loss: " + str(loss.item())) 
    else:
      # Recursive call
      self.__solve(lvl+1, data, target, v_H, nrelax_pre, nrelax_post, nrelax_coarse,
                   mgopt_printlevel, mgopt_levels, restrict_params, restrict_grads,
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
      e_h = get_interp_params(coarse_model, **interp_params_kwargs) 
      do_interp_states(model, coarse_model, **interp_states_kwargs)
  
    # 8. apply linesearch to update x_h
    #  x_h = x_h + alpha*e_h
    with torch.no_grad():
      x_h = get_params(model, deep_copy=False, grad=False)
      e_dot_gradf = tensor_list_dot(e_h, g_h).item()
      # Note, that line_search can store values between runs, like alpha, by having ls_kwargs = { 'ls_params' : {'alpha' : ...}} 
      do_line_search(lvl, e_h, x_h, v_h, model, optimizer, data, target, criterion, criterion_kwargs, compose, fine_loss, e_dot_gradf, mgopt_printlevel, **ls_kwargs)

    # 9. post-relaxation
    for k in range(nrelax_post):
      loss = compute_fwd_bwd_pass(lvl, optimizer, model, data, target, criterion, criterion_kwargs, compose, v_h)
      if (k==0):  root_print(rank, mgopt_printlevel, 2, "  CG Corr done loss:    " + str(loss.item()) ) 
      else:       root_print(rank, mgopt_printlevel, 2, "  Post-relax loss:      " + str(loss.item()) )
      optimizer.step()
    ##
    if (mgopt_printlevel == 2):  
      loss = compute_fwd_bwd_pass(lvl, optimizer, model, data, target, criterion, criterion_kwargs, compose, v_h)
      root_print(rank, mgopt_printlevel, 2, "  Post-relax loss:      " + str(loss.item()) )
  
    return loss.item()   
        
        

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
        """
        if isinstance(to_levelize, tuple) or isinstance(to_levelize, str):
            to_levelize = [to_levelize for i in range(max_levels)]
        elif isinstance(to_levelize, list):
            if len(to_levelize) < max_levels:
                mlz = max_levels - len(to_levelize)
                toext = [to_levelize[-1] for i in range(mlz)]
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
    if method == "tb_mgopt_cross_ent":
      criterion = tb_mgopt_cross_ent
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
    elif method == "no_line_search":
      # Do no line-search, just return x_h <--  x_h + e_h
      def line_search(lvl, e_h, x_h, v_h, model, optimizer, data, target, criterion, criterion_kwargs, compose, old_loss, e_dot_gradf, mgopt_printlevel, **ls_params):
        try: alpha = ls_params['alpha']
        except: alpha = 1.0
        tensor_list_AXPY(alpha, e_h, 1.0, x_h, inplace=True)
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
      

