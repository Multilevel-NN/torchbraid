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

###
# Examples to compare TB+NI vs. TB+MG/Opt vs. TB+MG/Opt+Local
###
#
# TB+NI
# $ python3 main_mgopt.py --steps 8 --samp-ratio 0.1 --lp-fwd-cfactor 2 --lp-bwd-cfactor 2 --mgopt-iter 0
#   ...
#   Train Epoch: 2 [4000/5000 (80%)]     	Loss: 1.323594	Time Per Batch 0.804948
#   Train Epoch: 2 [4500/5000 (90%)]     	Loss: 1.725507	Time Per Batch 0.803285
#   Train Epoch: 2 [5000/5000 (100%)]     	Loss: 1.746107	Time Per Batch 0.802796
#   Test set: Average loss: 0.0335, Accuracy: 380/1000 (38%)
#
# TB+MG/Opt (Takes the above NI solver and adds 1 epoch of MG/Opt)
# $ python3 main_mgopt.py --steps 8 --samp-ratio 0.1 --epochs 1 --mgopt-printlevel 2 --mgopt-levels 2 --mgopt-nrelax-pre 2 --mgopt-nrelax-post 2 --lp-fwd-cfactor 2 --lp-bwd-cfactor 2 
#   ...
#   ------------------------------------------------------------------------------
#   Train Epoch: 1 [5000/5000 (100%)]     	Loss: 2.450962	Time Per Batch 7.674536
#   ------------------------------------------------------------------------------
#   
#     Test set: Average loss: 0.0471, Accuracy: 103/1000 (10%)
#   
#     Test accuracy information for level 0
#       Test set: Average loss: 0.0461, Accuracy: 90/1000 (9%)
#   
#     Test accuracy information for level 1
#       Test set: Average loss: 0.0366, Accuracy: 305/1000 (30%)
#
# TB+MG/Opt+Local (Takes the above NI solver and adds 1 epoch of MG/Opt with purely local relaxation on each level)
# $ python3 main_mgopt.py --steps 8 --samp-ratio 0.1 --epochs 2 --mgopt-printlevel 2 --mgopt-levels 2 --mgopt-nrelax-pre 2 --mgopt-nrelax-post 2 --lp-fwd-cfactor 2 --lp-bwd-cfactor 2 --lp-fwd-levels 1 --lp-bwd-levels 1 --lp-fwd-finefcf --lp-bwd-finefcf --lp-fwd-relaxonlycg --lp-bwd-relaxonlycg --lp-fwd-finalrelax --lp-iters 1
#   ...
#   ------------------------------------------------------------------------------
#   Train Epoch: 2 [5000/5000 (100%)]     	Loss: 1.892899	Time Per Batch 3.777171
#   ------------------------------------------------------------------------------
#   
#    Test set: Average loss: 0.0498, Accuracy: 98/1000 (10%)
#   
#    Test accuracy information for level 0
#      Test set: Average loss: 0.0461, Accuracy: 89/1000 (9%)
#

from __future__ import print_function
import numpy as np

import torch
from torchvision import datasets, transforms
from mpi4py import MPI
from mgopt import parse_args, mgopt_solver

def main():
  
  ##
  # Parse command line args (function defined above)
  args = parse_args()
  procs = MPI.COMM_WORLD.Get_size()
  rank  = MPI.COMM_WORLD.Get_rank()
  
  ##
  # Load training and testing data 
  transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
  train_set = datasets.CIFAR10('./data', download=False,
                                   transform=transform,train=True)
  test_set  = datasets.CIFAR10('./data', download=False,
                                   transform=transform,train=False)
  
  ##
  # Define the train and test loaders, while reducing the number of samples (if desired) for faster execution
  train_set = torch.utils.data.Subset(train_set,range(int(50000*args.samp_ratio)))
  test_set = torch.utils.data.Subset(test_set,range(int(10000*args.samp_ratio)))
  train_loader = torch.utils.data.DataLoader(train_set,
                                             batch_size=args.batch_size, 
                                             shuffle=False)
  test_loader  = torch.utils.data.DataLoader(test_set,
                                             batch_size=args.batch_size, 
                                             shuffle=False)
  print("\nTraining setup:  Batch size:  " + str(args.batch_size) + "  Sample ratio:  " + str(args.samp_ratio) + "  MG/Opt Epochs:  " + str(args.epochs) )
  
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
    networks.append( ('ParallelNet', {'channels'          : args.channels, 
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
  optims = [ ("pytorch_sgd", { 'lr':args.lr, 'momentum':0.9}) for i in range(len(ni_steps)) ]

  ##
  # Initialize MG/Opt solver with nested iteration 
  epochs = 2
  mgopt_printlevel = 1
  log_interval = args.log_interval
  mgopt = mgopt_solver()
  mgopt.initialize_with_nested_iteration(ni_steps, train_loader, test_loader,
          networks, epochs=epochs, log_interval=log_interval,
          mgopt_printlevel=mgopt_printlevel, optims=optims, seed=args.seed) 
   
  print(mgopt)
  mgopt.options_used()
  

  ##
  # Run the MG/Opt solver
  #   Note: that we use the default restrict, interp, and line_search options, but these can be modified on a per-level basis
  if( args.mgopt_iter > 0):
    epochs = args.epochs
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
            mgopt_printlevel=mgopt_printlevel, mgopt_levels=mgopt_levels)
   
    print(mgopt)
    mgopt.options_used()
  ##
  

if __name__ == '__main__':
  main()



