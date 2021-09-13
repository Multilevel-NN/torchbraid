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

import torch
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

from mpi4py import MPI
from mgopt import parse_args, mgopt_solver

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
  print("\nTraining setup:  Batch size:  " + str(args.batch_size) + "  Sample ratio:  " + str(args.samp_ratio) 
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
    networks.append( ('ParallelNet', {'width'             : args.width, 
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

  ##
  # Initialize MG/Opt solver with nested iteration 
  epochs = 2
  mgopt_printlevel = 1
  log_interval = args.log_interval
  mgopt = mgopt_solver()
  mgopt.initialize_with_nested_iteration(ni_steps, train_loader, test_loader,
          networks, epochs=epochs, log_interval=log_interval,
          mgopt_printlevel=mgopt_printlevel, optims=optims, seed=args.seed,criterions="tb_mgopt_regression") 
   
  print(mgopt)
  mgopt.options_used()

 # Can change MGRIT options from NI to MG/Opt with the following
 #mgopt.levels[0].model.parallel_nn.setFwdNumRelax(0,level=0) 
  
  ##
  # Run the MG/Opt solver
  #   Note: that we use the default restrict and interp options, but these can be modified on a per-level basis
  if( args.mgopt_iter > 0):
    epochs = args.epochs
    line_search = ('no_line_search', {'a' : 1.0})
    log_interval = args.log_interval
    mgopt_printlevel = args.mgopt_printlevel
    mgopt_iter = args.mgopt_iter
    mgopt_levels = args.mgopt_levels
    mgopt_tol=0
    nrelax_pre = args.mgopt_nrelax_pre
    nrelax_post = args.mgopt_nrelax_post
    nrelax_coarse = args.mgopt_nrelax_coarse
    mgopt.mgopt_solve(train_loader, test_loader, epochs=epochs,
            log_interval=log_interval, mgopt_tol=mgopt_tol,
            mgopt_iter=mgopt_iter, nrelax_pre=nrelax_pre,
            nrelax_post=nrelax_post, nrelax_coarse=nrelax_coarse,
            mgopt_printlevel=mgopt_printlevel, mgopt_levels=mgopt_levels,
            line_search=line_search)
   
    print(mgopt)
    mgopt.options_used()
  ##

  with torch.no_grad():
    x = torch.unsqueeze(torch.linspace(-1.0,1.0,1001),1)
    trained_model = mgopt.levels[0].model

    plt.figure()
    plt.plot(x,trained_model(x))
    plt.plot(x,func(x))

  plt.show()

if __name__ == '__main__':
  main()
