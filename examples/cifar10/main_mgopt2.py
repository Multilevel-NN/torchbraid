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

# some helpful examples
# 
# BATCH_SIZE=50
# STEPS=12
# CHANNELS=8

# IN SERIAL
# python  main_mgopt2.py --steps ${STEPS} --channels ${CHANNELS} --batch-size ${BATCH_SIZE} --log-interval 100 --epochs 20 # 2>&1 | tee serial.out
# mpirun -n 4 python  main_mgopt2.py --steps ${STEPS} --channels ${CHANNELS} --batch-size ${BATCH_SIZE} --log-interval 100 --epochs 20 # 2>&1 | tee serial.out

from __future__ import print_function
import sys
import argparse
import statistics as stats
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchbraid
import torchbraid.utils

from torchvision import datasets, transforms

from timeit import default_timer as timer

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
  # reduce the number of samples for faster execution
  train_set = torch.utils.data.Subset(train_set,range(int(50000*args.samp_ratio)))
  test_set = torch.utils.data.Subset(test_set,range(int(10000*args.samp_ratio)))

  train_loader = torch.utils.data.DataLoader(train_set,
                                             batch_size=args.batch_size, 
                                             shuffle=True)
  test_loader  = torch.utils.data.DataLoader(test_set,
                                             batch_size=args.batch_size, 
                                             shuffle=False)
  
  ##
  # Compute number of nested iteration steps
  ni_steps = np.array([int(args.steps/(args.ni_rfactor**(args.ni_levels-i-1))) for i in range(args.ni_levels)])
  ni_steps = ni_steps[ ni_steps != 0 ]
  local_ni_steps = np.array(ni_steps / procs, dtype=int)

  ##
  # Define ParNet parameters for each nested iteration level, starting from coarse to fine
  networks = [] 
  for lsteps in local_ni_steps: 
    networks.append( ('ParallelNet', {'channels'    : args.channels, 
                                      'local_steps' : lsteps,
                                      'max_levels'  : args.lp_levels,
                                      'max_iters'   : args.lp_iters,
                                      'print_level' : args.lp_print} ) )
                                 
  ##
  # Define interpolation, optimization, and criterion for each nested iteration level (use the same at each level)
  interp_params = [ "tb_injection_interp_params" for i in range(len(ni_steps)) ]
  optims = [ ("pytorch_sgd", { 'lr':args.lr, 'momentum':0.9}) for i in range(len(ni_steps)) ]
  criterions = [ "tb_mgopt_cross_ent" for i in range(len(ni_steps)) ]


  ##
  # Initialize MG/Opt solver with nested iteration
  mgopt = mgopt_solver()
  training_setup = (train_loader, test_loader, args.samp_ratio, args.batch_size, args.epochs, args.log_interval)
  mgopt.initialize_with_nested_iteration(ni_steps, training_setup=training_setup, networks=networks, 
                                         interp_params=interp_params, optims=optims, criterions=criterions, 
                                         seed=args.seed)
  print(mgopt)
  print(mgopt.options_used())
  
  ##
  # Run the MG/Opt solver
  #   Note: the MG/Opt solver reuses the same interp_params, optims, and 
  #   criterions defined above for nested iteration. Thus, we only need to define a few more
  #   interp and restrict options here.  (This reuse could be changed, if needed.)
  
  ##
  # Define restriction strategy for network parameters, gradients, and state (use the same at each level)
  restrict_params = [ ("tb_get_injection_restrict_params", {'grad' : False}) for i in range(len(ni_steps)) ]
  restrict_grads  = [ ("tb_get_injection_restrict_params", {'grad' : True})  for i in range(len(ni_steps)) ]
  restrict_states = [ "tb_injection_restrict_network_state" for i in range(len(ni_steps)) ]
  interp_states   = [ "tb_injection_interp_network_state"   for i in range(len(ni_steps)) ]
  line_search     = [ ("tb_simple_backtrack_ls", {'n_line_search' : 6, 'alpha' : 1.0}) for i in range(len(ni_steps)) ]

  ##
  # Run MG/Opt cycles
  import pdb; pdb.set_trace()
  maxiter=2
  epochs=1 
  mgopt_tol=0 
  nrelax_pre=1
  nrelax_post=1 
  nrelax_coarse=5 
  mgopt_printlevel=1
  mgopt.mgopt_solve(self, maxiter=maxiter, epochs=epochs, mgopt_tol=mgopt_tol,
          nrelax_pre=nrelax_pre, nrelax_post=nrelax_post,
          nrelax_coarse=nrelax_coarse, mgopt_printlevel=mgopt_printlevel,
          restrict_params=restrict_params, restrict_grads=restrict_grads,
          restrict_states=restrict_states, interp_states=interp_states,
          line_search=line_search)
 

if __name__ == '__main__':
  main()



