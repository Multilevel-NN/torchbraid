#@HEADER
# ************************************************************************
# 
#                        Torchbraid v. 0.1
# 
# Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC 
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. 
# Government retains certain rights in this software.
# 
# Torchbraid is licensed under 3-clause BSD terms of use:
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
# 
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
# 
# 3. Neither the name National Technology & Engineering Solutions of Sandia, 
# LLC nor the names of the contributors may be used to endorse or promote 
# products derived from this software without specific prior written permission.
# 
# Questions? Contact Eric C. Cyr (eccyr@sandia.gov)
# 
# ************************************************************************
#@HEADER

from __future__ import print_function
import numpy as np
import argparse

import torch
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

from mpi4py import MPI
from torchbraid.mgopt import mgopt_solver

import network

class RegressionDataset(Dataset):
  def __init__(self,x,func):
    # pad out array to have a domain dimension of at least 1
    if len(x.shape)==1:
      x = torch.unsqueeze(x,1)      

    self.x = x
    self.func = func

    self.domain_dim = self.x.shape[1]
    self.range_dim = func(self.x[0:2]).shape[1] # this gets the shape of the range

  def domainDim(self):
    return self.domain_dim

  def rangeDim(self):
    return self.range_dim
    
  def __len__(self):
    return self.x.shape[0]

  def __getitem__(self,idx):
    return self.x[idx], self.func(self.x[idx])

####################################################################################
####################################################################################
# Parsing functions

def parse_args():
  """
  Return back an args dictionary based on a standard parsing of the command line inputs
  """
  
  # Command line settings
  parser = argparse.ArgumentParser(description='MG/Opt Solver Parameters')
  parser.add_argument('--seed', type=int, default=1, metavar='S',
                      help='random seed (default: 1)')
  parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                      help='how many batches to wait before logging training status')
  
  # artichtectural settings
  parser.add_argument('--steps', type=int, default=4, metavar='N',
                      help='Number of times steps in the resnet layer (default: 4)')
  parser.add_argument('--channels', type=int, default=8, metavar='N',
                      help='Number of channels in resnet layer (default: 8)')
  parser.add_argument('--tf',type=float,default=1.0,
                      help='Final time')
  parser.add_argument('--width',type=int,default=32,
                      help='Network width')

  # algorithmic settings (gradient descent and batching)
  parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                      help='input batch size for training (default: 50)')
  parser.add_argument('--epochs', type=int, default=2, metavar='N',
                      help='number of epochs to train (default: 2)')
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

def main():
  
  ##
  # Parse command line args (function defined above)
  args = parse_args()
  procs = MPI.COMM_WORLD.Get_size()
  rank  = MPI.COMM_WORLD.Get_rank()

  ## 
  # Fix the seed if selected, otherwise let it be random
  if args.seed>0:
    torch.manual_seed(args.seed)
  else:
    args.seed = torch.seed()

  print(f'Initial seed used = {args.seed}')
  
  
  ##
  # Load training and testing data, while reducing the number of samples (if desired) for faster execution

  num_train_points = 1000
  num_test_points = 100
 
  omega = 4.0
  func = lambda x: torch.sin(omega*np.pi*x)
  x_train = 2.0*torch.rand((num_train_points,))-1.0
  x_test = 2.0*torch.rand((num_test_points,))-1.0

  #
  train_set = RegressionDataset(x_train,func)
  test_set = RegressionDataset(x_test,func)
  #
  train_loader = torch.utils.data.DataLoader(train_set,batch_size=args.batch_size,shuffle=False)
  test_loader = torch.utils.data.DataLoader(test_set,batch_size=args.batch_size,shuffle=False)
  print("\nTraining setup:  Batch size:  " + str(args.batch_size)  
                       + "  MG/Opt Epochs:  " + str(args.epochs) )
  
  ##
  # Compute number of nested iteration steps, going from fine to coarse
  ni_steps = np.array([int(args.steps/(args.ni_rfactor**(args.ni_levels-i-1))) for i in range(args.ni_levels)])
  ni_steps = ni_steps[ ni_steps != 0 ]
  local_ni_steps = np.flip( np.array(ni_steps / procs, dtype=int) )
  print("\nNested iteration steps:  " + str(ni_steps))

  ##
  # Define ParNet parameters for each nested iteration level, starting from fine to coarse
  networks = [] 
  for lsteps in local_ni_steps: 
    networks.append( ('Factory',
                     {'width'             : args.width, 
                      'input_size'        : train_set.domainDim(),
                      'output_size'       : train_set.rangeDim(),
                      'local_steps'       : lsteps,
                      'max_iters'         : args.lp_iters,
                      'print_level'       : args.lp_print,
                      'Tf'                : args.tf,
                      'max_fwd_levels'    : args.lp_fwd_levels,
                      'max_bwd_levels'    : args.lp_bwd_levels,
                      'max_fwd_iters'     : args.lp_fwd_iters,
                      'print_level'       : args.lp_print,
                      'braid_print_level' : args.lp_braid_print,
                      'fwd_cfactor'       : args.lp_fwd_cfactor,
                      'bwd_cfactor'       : args.lp_bwd_cfactor,
                      'fine_fwd_fcf'      : args.lp_fwd_finefcf,
                      'fine_bwd_fcf'      : args.lp_bwd_finefcf,
                      'fwd_nrelax'        : args.lp_fwd_nrelax_coarse,
                      'bwd_nrelax'        : args.lp_bwd_nrelax_coarse,
                      'skip_downcycle'    : not args.lp_use_downcycle,
                      'fmg'               : args.lp_use_fmg,
                      'fwd_relax_only_cg' : args.lp_fwd_relaxonlycg,
                      'bwd_relax_only_cg' : args.lp_bwd_relaxonlycg,
                      'CWt'               : args.lp_use_crelax_wt,
                      'fwd_finalrelax'    : args.lp_fwd_finalrelax
                      }))
                                 
  ##
  # Specify optimization routine on each level, starting from fine to coarse
  optims = [ ("pytorch_adam", { 'lr': args.lr}) for i in range(len(ni_steps)) ]

  model_factory = lambda level,**kwargs: network.ParallelNet(**kwargs)

  ##
  # Initialize MG/Opt solver with nested iteration 
  epochs = args.NIepochs
  mgopt_printlevel = 1
  log_interval = args.log_interval
  mgopt = mgopt_solver()
  mgopt.initialize_with_nested_iteration(model_factory,
          ni_steps, train_loader, test_loader,
          networks, epochs=epochs, log_interval=log_interval,
          mgopt_printlevel=mgopt_printlevel, optims=optims, seed=args.seed,criterions="tb_mgopt_regression") 
   
  print(mgopt)
  mgopt.options_used()

  ##
  # Run the MG/Opt solver
  #   Note: that we use the default restrict and interp options, but these can be modified on a per-level basis
  if( args.mgopt_iter > 0):
    epochs = args.epochs
    line_search = ('tb_simple_ls', {'ls_params' : {'alphas' : [0.01, 0.1, 0.5, 1.0, 2.0, 4.0]}} )
    line_search = ('tb_simple_backtrack_ls', {'ls_params' : {'n_line_search' : 10, 'alpha' : 1.0, 'c1' : 1e-4}} )
    log_interval = args.log_interval
    mgopt_printlevel = args.mgopt_printlevel
    mgopt_iter = args.mgopt_iter
    mgopt_levels = args.mgopt_levels
    mgopt_tol=0
    nrelax_pre = args.mgopt_nrelax_pre
    nrelax_post = args.mgopt_nrelax_post
    nrelax_coarse = args.mgopt_nrelax_coarse
    loss = mgopt.mgopt_solve(train_loader, test_loader, epochs=epochs,
                             log_interval=log_interval, mgopt_tol=mgopt_tol,
                             mgopt_iter=mgopt_iter, nrelax_pre=nrelax_pre,
                             nrelax_post=nrelax_post, nrelax_coarse=nrelax_coarse,
                             mgopt_printlevel=mgopt_printlevel, mgopt_levels=mgopt_levels,
                             line_search=line_search)

    print(mgopt)
    mgopt.options_used()
   
  f,ax = plt.subplots(2,len(mgopt.levels),figsize=(12,9))
  with torch.no_grad():
    x = torch.unsqueeze(torch.linspace(-1.0,1.0,1000),1)

    ax[0,0].plot(x,func(x),label='exact')
    ax[0,0].plot(x,mgopt.levels[0].model(x),label=f'level=0')
    ax[0,0].legend()

  ax[0,1].plot(loss)

  for l,lvl in enumerate(mgopt.levels):
    ax[1,l].plot(lvl.out_ls_step,'.',label=f'level={l},cnt={len(lvl.out_ls_step)}')
    ax[1,l].set_title(f'ls level {l}')

  plt.show()

if __name__ == '__main__':
  main()
