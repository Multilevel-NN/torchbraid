from __future__ import print_function

import numpy as np

import sys
import statistics as stats
import argparse
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchbraid
import torchbraid.utils
from torchbraid.utils import LPBatchNorm2d

from torchbraid.mgopt import root_print, compute_levels

from timeit import default_timer as timer

from torchbraid.utils import MPI, git_rev, getDevice

__all__ = [ 'parse_args', 'ParallelNet' ]

# Related to:
# https://arxiv.org/pdf/2002.09779.pdf

def getComm():
  return MPI.COMM_WORLD

def batch_norm(lp_batch_norm,*args,**kwargs):
  if lp_batch_norm:
    return LPBatchNorm2d(*args,**kwargs)
  else:
    return nn.BatchNorm2d(*args,**kwargs)

####################################################################################
####################################################################################
# Classes and functions that define the basic network types for MG/Opt and TorchBraid.
class OpenLayer(nn.Module):
  ''' Opening layer (not ODE-net, not parallelized in time) '''
  def __init__(self,channels,seed,lp_batch_norm=False):
    super(OpenLayer, self).__init__()

    torch.manual_seed(seed)

    self.channels = channels
    self.pre = nn.Sequential(
      nn.Conv2d(3, channels, kernel_size=7, padding=3, stride=2,bias=False),
      batch_norm(lp_batch_norm,channels,momentum=0.1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    )

  def forward(self, x):
    return self.pre(x)
# end layer

class CloseLayer(nn.Module):
  ''' Closing layer (not ODE-net, not parallelized in time) '''
  def __init__(self,in_channels,seed):
    super(CloseLayer, self).__init__()

    torch.manual_seed(seed)

    self.avg  = nn.AdaptiveAvgPool2d(1)
    self.fc   = nn.Linear(in_channels, 200)

  def forward(self, x):
    x = self.avg(x)
    out = torch.flatten(x.view(x.size(0),-1), 1)
    return self.fc(out)
# end layer


class TransitionLayer(nn.Module):
  ''' Closing layer (not ODE-net, not parallelized in time) '''
  def __init__(self,channels,seed,pooling=False, keep_all_features=False,lp_batch_norm=False):
    super(TransitionLayer, self).__init__()

    torch.manual_seed(seed)

    out_channels = 2*channels if not keep_all_features else 4*channels

    # Account for 64x64 image and 3 RGB channels
    if pooling:
      self.pre = nn.Sequential(
        nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
        nn.Conv2d(channels, out_channels, kernel_size=3, stride=2, padding=1,bias=False),
        batch_norm(lp_batch_norm,out_channels),
        nn.ReLU()
      )
    else:
      self.pre = nn.Sequential(
        nn.Conv2d(channels, out_channels, kernel_size=3, stride=2, padding=1,bias=False),
        batch_norm(lp_batch_norm,out_channels),
        nn.ReLU()
      )

  def forward(self, x):
    return self.pre(x)
# end layer

class StepLayer(nn.Module):
  ''' Single ODE-net layer will be parallelized in time ''' 
  def __init__(self,channels,activation='relu',seed=1,lp_batch_norm=False):
    super(StepLayer, self).__init__()
    ker_width = 3

    torch.manual_seed(seed)
    
    # Account for 3 RGB Channels
    self.conv1 = nn.Conv2d(channels,channels,ker_width,padding=1,bias=False)
    self.bn1   = batch_norm(lp_batch_norm,channels, momentum=0.1)
    self.conv2 = nn.Conv2d(channels,channels,ker_width,padding=1,bias=False)
    self.bn2   = batch_norm(lp_batch_norm,channels, momentum=0.1)

    self.activation1 = nn.ReLU()
    self.activation2 = nn.ReLU()

  def forward(self, x):
    y = self.conv1(x) 
    y = self.bn1(y) 
    y = self.activation1(y)
    y = self.conv2(y) 
    y = self.bn2(y) 
    y = self.activation2(y)
    return y 
# end layer

def buildNet(parallel,**network):
  if parallel:
    m = ParallelNet(**network)
  else:
    m = SerialNet(**network)
  m.construct_params = network
  return m
# end buildNet

def buildNetClone(parallel,src,replace):
  new_params = {**src.construct_params}
  new_params.update(replace)

  if parallel:
    return ParallelNet(**new_params)
  else:
    return SerialNet(**new_params)

def remove_self(m):
  m.pop('self')
  return m

class SerialNet(nn.Module):
  class ODEBlock(nn.Module):
    """This is a helper class to wrap layers that should be ODE time steps."""
    def __init__(self,dt,layer):
      # save the constructor parameters (helpful for building coarse grids if desired)
      self.construct_params = remove_self(locals())

      super(SerialNet.ODEBlock, self).__init__()

      self.layer = layer
      self.dt = dt

    def forward(self, x):
      y = self.dt*self.layer(x)
      y.add_(x)
      return y
  # end ODEBlock

  def __init__(self, channels=8, global_steps=8, Tf=1.0, max_fwd_levels=1, max_bwd_levels=1, max_iters=1, max_fwd_iters=0, 
                     print_level=0, braid_print_level=0, fwd_cfactor=4, bwd_cfactor=4, fine_fwd_fcf=False, 
                     fine_bwd_fcf=False, fwd_nrelax=1, bwd_nrelax=1, skip_fwd_downcycle=True, skip_bwd_downcycle=True, fmg=False, fwd_relax_only_cg=0, 
                     bwd_relax_only_cg=0, CWt=1.0, fwd_finalrelax=False,diff_scale=0.0,activation='tanh', coarse_frelax_only=False,
                     fwd_crelax_wt=1.0, pooling=False, seed=1, keep_all_features=False):
    super(SerialNet, self).__init__()

    step_layer_1 = lambda: StepLayer(channels,seed)
    step_layer_2 = lambda: (StepLayer(2*channels,seed) if not keep_all_features else StepLayer(4*channels, seed))
    step_layer_3 = lambda: (StepLayer(4*channels,seed) if not keep_all_features else StepLayer(16*channels, seed))
    step_layer_4 = lambda: (StepLayer(8*channels,seed) if not keep_all_features else StepLayer(64*channels, seed))

    open_layer    = lambda: OpenLayer(channels,seed)
    trans_layer_1 = lambda: TransitionLayer(channels,seed, pooling, keep_all_features)
    trans_layer_2 = lambda: (TransitionLayer(2*channels,seed, pooling, keep_all_features) if not keep_all_features else TransitionLayer(4*channels, seed, pooling, keep_all_features))
    trans_layer_3 = lambda: (TransitionLayer(4*channels,seed, pooling, keep_all_features) if not keep_all_features else TransitionLayer(16*channels, seed, pooling, keep_all_features))

    layers_temp = [open_layer,    step_layer_1, trans_layer_1,    step_layer_2,  trans_layer_2,step_layer_3, trans_layer_3, step_layer_4]
    num_steps   = [         1,  global_steps-1,             1,   global_steps-1, 1,            global_steps-1,           1, global_steps-1]

    layers = []
    dt = Tf/sum(num_steps)
    for l,n in zip(layers_temp,num_steps):
      if n>1:
        layers += [SerialNet.ODEBlock(dt,l()) for i in range(n)]
      else:
        layers+= [l()]
    layers += [CloseLayer(8*channels,seed)] if not keep_all_features else [CloseLayer(64*channels,seed)]

    self.serial_nn = nn.Sequential(*layers)

    # this object ensures that only the LayerParallel code runs on ranks!=0
    self.compose = lambda op,*p,**k: op(*p,**k)

  def forward(self, x):
    # by passing this through 'o' (mean composition: e.g. self.open_nn o x) 
    # this makes sure this is run on only processor 0

    x = self.serial_nn(x)

    return x

  def getFwdStats(self):
    return 1,0.0
    
  def getBwdStats(self):
    return 1,0.0

  def saveParams(self,rank,model_dir):
    torch.save(self.state_dict(),f'{model_dir}/serial_model.{rank}.mdl')

  def loadParams(self,rank,model_dir):
    self.load_state_dict(torch.load(f'{model_dir}/serial_model.{rank}.mdl',weights_only=True))

  def startStateCommunication(self):
    pass # no-op


class ParallelNet(nn.Module):
  ''' Full parallel ODE-net based on StepLayer,  will be parallelized in time ''' 
  def __init__(self, channels=8, global_steps=8, Tf=1.0, max_fwd_levels=1, max_bwd_levels=1, max_iters=1, max_fwd_iters=0, 
                     print_level=0, braid_print_level=0, fwd_cfactor=4, bwd_cfactor=4, fine_fwd_fcf=False, fine_bwd_fcf=False, 
                     fwd_nrelax=1, bwd_nrelax=1, skip_fwd_downcycle=True, skip_bwd_downcycle=True, fmg=False, fwd_relax_only_cg=0, bwd_relax_only_cg=0, 
                     CWt=1.0, fwd_crelax_wt=1.0, fwd_finalrelax=False,diff_scale=0.0, activation='tanh', 
                     coarse_frelax_only=False, pooling=False, seed=1, keep_all_features=False):
    super(ParallelNet, self).__init__()

    step_layer_1 = lambda: StepLayer(channels,seed,lp_batch_norm=True)
    step_layer_2 = lambda: (StepLayer(2*channels,seed,lp_batch_norm=True) if not keep_all_features else StepLayer(4*channels, seed,lp_batch_norm=True))
    step_layer_3 = lambda: (StepLayer(4*channels,seed,lp_batch_norm=True) if not keep_all_features else StepLayer(16*channels, seed,lp_batch_norm=True))
    step_layer_4 = lambda: (StepLayer(8*channels,seed,lp_batch_norm=True) if not keep_all_features else StepLayer(64*channels, seed,lp_batch_norm=True))

    open_layer = lambda: OpenLayer(channels,seed,lp_batch_norm=True)
    trans_layer_1 = lambda: TransitionLayer(channels,seed, pooling, keep_all_features,lp_batch_norm=True)
    trans_layer_2 = lambda: (TransitionLayer(2*channels,seed, pooling, keep_all_features,lp_batch_norm=True) if not keep_all_features else TransitionLayer(4*channels, seed, pooling, keep_all_features,lp_batch_norm=True))
    trans_layer_3 = lambda: (TransitionLayer(4*channels,seed, pooling, keep_all_features,lp_batch_norm=True) if not keep_all_features else TransitionLayer(16*channels, seed, pooling, keep_all_features,lp_batch_norm=True))

    layers    = [open_layer,    step_layer_1, trans_layer_1,    step_layer_2,  trans_layer_2,step_layer_3, trans_layer_3, step_layer_4]
    num_steps = [         1,  global_steps-1,             1,   global_steps-1, 1,            global_steps-1,           1, global_steps-1]

    self.parallel_nn = torchbraid.LayerParallel(MPI.COMM_WORLD,layers,num_steps,Tf,max_fwd_levels=max_fwd_levels,max_bwd_levels=max_bwd_levels,max_iters=max_iters)
    if max_fwd_iters>0:
      self.parallel_nn.setFwdMaxIters(max_fwd_iters)
    self.parallel_nn.setPrintLevel(print_level,True)
    self.parallel_nn.setPrintLevel(braid_print_level,False)
    self.parallel_nn.setFwdCFactor(fwd_cfactor)
    self.parallel_nn.setBwdCFactor(bwd_cfactor)
    self.parallel_nn.setSkipFwdDowncycle(skip_fwd_downcycle)
    self.parallel_nn.setSkipBwdDowncycle(skip_bwd_downcycle)
    self.parallel_nn.setBwdRelaxOnlyCG(bwd_relax_only_cg)
    self.parallel_nn.setFwdRelaxOnlyCG(fwd_relax_only_cg)
    self.parallel_nn.setCRelaxWt(CWt)
    self.parallel_nn.setFwdCRelaxWt(fwd_crelax_wt)
    self.parallel_nn.setMinCoarse(2)
    self.parallel_nn.setFwdResidualCompute(False)
    self.parallel_nn.setBwdResidualCompute(False)

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
    
    if coarse_frelax_only:
      self.parallel_nn.setNumRelax(0)
      self.parallel_nn.setNumRelax(0, level=0)

    # this object ensures that only the LayerParallel code runs on ranks!=0
    self.compose = self.parallel_nn.comp_op()

    # by passing this through 'compose' (mean composition: e.g. OpenLayer o channels) 
    # on processors not equal to 0, these will be None (there are no parameters to train there)
    self.close_nn = self.compose(CloseLayer,8*channels,seed) if not keep_all_features else self.compose(CloseLayer, 64*channels, seed)

  def __repr__(self):
    return self.parallel_nn.repr_helper(self)

  def forward(self, x):
    # by passing this through 'o' (mean composition: e.g. self.open_nn o x) 
    # this makes sure this is run on only processor 0

    x = self.parallel_nn(x)
    x = self.compose(self.close_nn,x)

    return x

  def getFwdStats(self):
    return self.parallel_nn.getFwdStats()
    
  def getBwdStats(self):
    return self.parallel_nn.getBwdStats()

  def saveParams(self,rank,model_dir):
    torch.save(self.state_dict(),f'{model_dir}/parallel_model.{rank}.mdl')

  def loadParams(self,rank,model_dir):
    self.load_state_dict(torch.load(f'{model_dir}/parallel_model.{rank}.mdl',weights_only=True))

  def startStateCommunication(self):
    self.parallel_nn.startStateCommunication() 

# end ParallelNet 
####################################################################################
####################################################################################


####################################################################################
####################################################################################
# Parsing functions

def parse_args(mgopt_on=True):
  """
  Return back an args dictionary based on a standard parsing of the command line inputs
  """
  
  # Command line settings
  parser = argparse.ArgumentParser(description='MG/Opt Solver Parameters')
  parser.add_argument('--seed', type=int, default=1, metavar='S',
                      help='random seed (default: 1)')
  parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                      help='how many batches to wait before logging training status')
  parser.add_argument('--use-serial',action='store_true', default=False, 
                      help='Turn on serial run (default: False)')

  # model saving and loading settings
  parser.add_argument('--save-model',action='store_true', default=False,
                      help='Save the model to disk after each epoch (default: False)')
  parser.add_argument('--load-model',action='store_true', default=False,
                      help='Load the model from disk at startup (default: False)')
  parser.add_argument('--load-optimizer',action='store_true', default=False,
                      help='Load the optimizer from disk at startup (default: False)')
  parser.add_argument('--load-lr-scheduler',action='store_true', default=False,
                      help='Load the LR scheduler from disk at startup (default: False)')
  parser.add_argument('--model-dir',default='./',
                      help='Location to Save the model to (default: ./)')
  
  # artichtectural settings
  parser.add_argument('--steps', type=int, default=4, metavar='N',
                      help='Number of times steps in the resnet layer (default: 4)')
  parser.add_argument('--channels', type=int, default=8, metavar='N',
                      help='Number of channels in resnet layer (default: 8)')
  parser.add_argument('--tf',type=float,default=1.0,
                      help='Final time')
  parser.add_argument('--diff-scale',type=float,default=0.0,
                      help='Diffusion coefficient')
  parser.add_argument('--activation',type=str,default='relu',
                      help='Activation function')
  parser.add_argument('--pooling', action='store_true', default=False,
                      help='Add an AvgPool2D to the transition layers')
  parser.add_argument('--keep-all-features', action='store_true', default=False,
                      help='Keep the number of features consistent when performing a convolution (default: False)')

  # algorithmic settings (gradient descent and batching)
  parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                      help='input batch size for training (default: 50)')
  parser.add_argument('--epochs', type=int, default=2, metavar='N',
                      help='number of epochs to train (default: 2)')
  parser.add_argument('--samp-ratio', type=float, default=1.0, metavar='N',
                      help='number of samples as a ratio of the total number of samples')
  parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                      help='learning rate (default: 0.01)')
  parser.add_argument('--lr-scheduler', nargs='*',
                      help='Turn on the learning rate scheduler')
  parser.add_argument('--opt', type=str, default='SGD',
                      help='Optimizer, SGD or Adam')

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
  parser.add_argument('--lp-fwd-cfactor', type=int, default=4, metavar='N', nargs='*',
                      help='Layer parallel coarsening factor for forward solve (default: 4)')
  parser.add_argument('--lp-bwd-cfactor', type=int, default=4, metavar='N', nargs='*',
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
                      help='Layer parallel use backward and forward downcycle on or off (default: False). Overrides the lp-use-fwd-downcycle and lp-use-bwd-downcycle options')
  parser.add_argument('--lp-use-bwd-downcycle',action='store_true', default=False, 
                      help='Layer parallel use backward downcycle on or off (default: False)')
  parser.add_argument('--lp-use-fwd-downcycle',action='store_true', default=False, 
                      help='Layer parallel use forward downcycle on or off (default: False)')
  parser.add_argument('--lp-use-fmg',action='store_true', default=False, 
                      help='Layer parallel use FMG for one cycle (default: False)')
  parser.add_argument('--lp-bwd-relaxonlycg',action='store_true', default=0, 
                      help='Layer parallel use relaxation only on coarse grid for backward cycle (default: False)')
  parser.add_argument('--lp-fwd-relaxonlycg',action='store_true', default=0, 
                      help='Layer parallel use relaxation only on coarse grid for forward cycle (default: False)')
  parser.add_argument('--lp-use-crelax-wt', type=float, default=1.0, metavar='CWt',
                      help='Layer parallel use weighted C-relaxation on backwards solve (default: 1.0).  Not used for coarsest braid level.')
  parser.add_argument('--lp-fwd-crelax-wt', type=float, default=1.0,
                      help='Layer parallel use weighted C-relatation on forwards solve (default: 1.0).  Not used for coarest braid level.')
  parser.add_argument('--lp-coarse-frelax-only', action='store_true', default=False,
                      help='Use F-relaxation only on the coarse grids.')

  # Mean Initial Guess Storage
  parser.add_argument('--mig-storage', default=None, nargs='?', const=0.1, type=float,
                      help='Use Mean Initial Guess Storage (default: False)')

  if mgopt_on:
    parser.add_argument('--NIepochs', type=int, default=2, metavar='N',
                        help='number of epochs per Nested Iteration (default: 2)')
  
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
  # end mgopt_on

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

  if len(args.lp_fwd_cfactor) == 0:
    root_print(rank, 1, 1, 'The lp_fwd_cfactor argument was given without a value, specify a value.')
    sys.exit(0)
  elif len(args.lp_fwd_cfactor) > 1:
    # build the dictionary to match levels with cfactor
    args.lp_fwd_cfactor = {i: cfact for i, cfact in enumerate(args.lp_fwd_cfactor)}
  else:
    args.lp_fwd_cfactor = args.lp_fwd_cfactor[0]

  
  if len(args.lp_bwd_cfactor) == 0:
    root_print(rank, 1, 1, 'The lp_bwd_cfactor argument was given without a value, specify a value.')
    sys.exit(0)
  elif len(args.lp_bwd_cfactor) > 1:
    # build the dictionary to match levels with cfactor
    args.lp_bwd_cfactor = {i: cfact for i, cfact in enumerate(args.lp_bwd_cfactor)}
  else:
    args.lp_bwd_cfactor = args.lp_bwd_cfactor[0]
  

  # lp_use_downcycle overrides the fwd and backward options
  if args.lp_use_downcycle:
    args.lp_use_fwd_downcycle = True
    args.lp_use_bwd_downcycle = True

  if mgopt_on:
    ni_levels = args.ni_levels
    ni_rfactor = args.ni_rfactor
    if args.steps % ni_rfactor**(ni_levels-1) != 0:
      root_print(rank, 1, 1, 'Steps combined with the coarsest nested iteration level must be an even multiple: %d %d %d' % (args.steps,ni_rfactor,ni_levels-1))
      sys.exit(0)
    
    if args.steps / ni_rfactor**(ni_levels-1) % procs != 0:
      root_print(rank, 1, 1, 'Coarsest nested iteration must fit on the number of processors')
      sys.exit(0)

  return args


def get_lr_scheduler(optimizer, args):
  if (len(args.lr_scheduler) == 0) or args.lr_scheduler[0] == 'step':
    step_size = 7
    gamma = 0.1
    if len(args.lr_scheduler) > 1:
      step_size = int(args.lr_scheduler[1])
    if len(args.lr_scheduler) > 2:
      gamma = float(args.lr_scheduler[2])
          
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=gamma)
  
  if args.lr_scheduler[0] == 'exp':
    gamma = 0.9
    if len(args.lr_scheduler) > 1:
      gamma = float(args.lr_scheduler[1])
        
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
  
  if args.lr_scheduler[0] == 'reduceonplateau':
    patience = 10
    factor = 0.1
    if len(args.lr_scheduler) > 1:
      patience = int(args.lr_scheduler[1])
    if len(args.lr_scheduler) > 2:
      factor = float(args.lr_scheduler[2])
            
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=patience, factor=factor)

  if args.lr_scheduler[0] == 'cycliclr':
    base_lr = 1e-4
    max_lr = 0.1
    step_size_up = 2000
    step_size_down = None
    if len(args.lr_scheduler) > 1:
      base_lr = float(args.lr_scheduler[1])
    if len(args.lr_scheduler) > 2:
      max_lr = float(args.lr_scheduler[2])
    if len(args.lr_scheduler) > 3:
      mode = args.lr_scheduler[3]
    if len(args.lr_scheduler) > 4:
      step_size_up = int(args.lr_scheduler[4])
    if len(args.lr_scheduler) > 5:
      step_size_down = int(args.lr_scheduler[5])

    return torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr, max_lr, step_size_up=step_size_up, step_size_down=step_size_down, mode=mode)
    
####################################################################################
####################################################################################
