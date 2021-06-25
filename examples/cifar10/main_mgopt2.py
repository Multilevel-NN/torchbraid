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
# python  main.py --steps ${STEPS} --channels ${CHANNELS} --batch-size ${BATCH_SIZE} --log-interval 100 --epochs 20 # 2>&1 | tee serial.out
# mpirun -n 4 python  main.py --steps ${STEPS} --channels ${CHANNELS} --batch-size ${BATCH_SIZE} --log-interval 100 --epochs 20 # 2>&1 | tee serial.out

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


## Do you need all these imports...?

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
  # Define interpolation for each nested iteration level (use the same interpolation at each level)
  interp_netw_params = [ "piecewise_const_interp_network_params" for i in range(len(ni_steps)) ]

  ##
  # Define optimization strategy for each nested iteration level (use the same at each level)
  optims = [ ("pytorch_sgd", { 'lr':args.lr, 'momentum':0.9}) for i in range(len(ni_steps)) ]

  ##
  # Define optimization strategy for each nested iteration level (use the same at each level)
  criterions = [ "basic_mgopt_cross_ent" for i in range(len(ni_steps)) ]

  ##
  # Initialize MG/Opt solver with nested iteration
  mgopt = mgopt_solver()
  training_setup = (train_loader, test_loader, args.samp_ratio, args.batch_size, args.epochs, args.log_interval)
  mgopt.initialize_with_nested_iteration(ni_steps, training_setup=training_setup, networks=networks, 
                                         interp_netw_params=interp_netw_params, optims=optims, 
                                         criterions=criterions, seed=args.seed)
  
  print(mgopt)
  print(mgopt.options_used())
  import pdb; pdb.set_trace()
  return
  
  # can you turn anything else into a dictionary, like training_setup?  Or just
  # leave as is?  Dictionary is better, better param and name checking.
  
  ############################
  #import pdb; pdb.set_trace()
  ############################

  # Now, carry out V-cycles.  Hierarchy is initialized.
  # First, reverse list, so entry 0 is the finest level
  #
  mgopt_iters = 1
  nrelax_pre = 1
  nrelax_post = 1
  nrelax_coarse = 5 # number of optimizations for coarse-grid solve
  cf = 2
  lr = args.lr
  momentum = 0.9
  
  # make train_loader a solve parameter
  # make optim.SGD a solve parameter

  n_line_search = 6
  alpha = 1.0
  
  mgopt_printlevel = 1
  log_interval = args.log_interval
  epochs = args.epochs
  #
  # likely recursive parameters 
  lvl = 0
  v_h = None
  #
  for epoch in range(1, epochs + 1):
    
    total_time = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
      
      start = timer()
      for it in range(mgopt_iters):
        if (mgopt_printlevel == 1) and (lvl == 0):  root_print(rank, "\nMG/Opt Iter:  " + str(it) + "   batch_idx:  " + str(batch_idx) ) 
        if mgopt_printlevel == 1:  root_print(rank, "\n  Level:  " + str(lvl)) 

        optimizer_fine = optim.SGD(models[lvl].parameters(), lr=lr, momentum=momentum)
        
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
          piecewise_const_restrict_network_state(models[lvl], models[lvl+1], cf=2)
          gtilde_H = piecewise_const_restrict_network_params(models[lvl], cf=cf, deep_copy=True, grad=True)
          x_H      = piecewise_const_restrict_network_params(models[lvl], cf=cf, deep_copy=True, grad=False)
          # For x_H to take effect, these parameters must be written to the next coarser network
          write_network_params_inplace(models[lvl+1], x_H)
          # Must store x_H for later error computation
          x_H_zero = tensor_list_deep_copy(x_H)
    
        # 4. compute gradient on coarse level, using restricted parameters
        #  g_H = grad( f_H(x_H) )
        optimizer_coarse = optim.SGD(models[lvl+1].parameters(), lr=lr, momentum=momentum)
        # Evaluate gradient.  For computing fwd_bwd_pass, give 0 as first
        # parameter, so that the MGOpt term is turned-off.  We just want hte
        # gradient of f_H here.
        loss = compute_fwd_bwd_pass(0, optimizer_coarse, models[lvl+1], data, target, my_criterion, compose, None)
        with torch.no_grad():
          g_H = get_network_params(models[lvl+1], deep_copy=True, grad=True)
    
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
          x_H = get_network_params(models[lvl+1], deep_copy=False, grad=False)
          e_H = tensor_list_AXPY(1.0, x_H, -1.0, x_H_zero)
          #
          # to correctly interpolate e_H --> e_h, we need to put these values in a
          # network, so the interpolation knows the layer-parallel structure and
          # where to refine.
          write_network_params_inplace(models[lvl+1], e_H)
          e_h = piecewise_const_interp_network_params(models[lvl+1], cf=cf, deep_copy=True, grad=False)
          piecewise_const_interp_network_state(models[lvl], models[lvl+1], cf=2)
          
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
      
      if batch_idx % log_interval == 0:
        if (mgopt_printlevel == 1):  root_print(rank, "\n------------------------------------------------------------------------")
        root_print(rank,'Train Epoch: {} [{}/{} ({:.0f}%)]     \tLoss: {:.6f}\tTime Per Batch {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item(), total_time/(batch_idx+1.0)))
        if (mgopt_printlevel == 1):  root_print(rank, "------------------------------------------------------------------------")
    
    ## End Batch loop

    root_print(rank,'Train Epoch: {} [{}/{} ({:.0f}%)]     \tLoss: {:.6f}\tTime Per Batch {:.6f}'.format(
      epoch, (batch_idx+1) * len(data), len(train_loader.dataset),
      100. * (batch_idx+1) / len(train_loader), loss.item(),total_time/(batch_idx+1.0)))

    ##
    # no need to return anything, fine network is modified in place

  # --> Just use these times from the MG/Opt, not the NI part
  root_print(rank,'TIME PER EPOCH: %.2e (1 std dev %.2e)' % (stats.mean(epoch_times),stats.stdev(epoch_times)))
  root_print(rank,'TIME PER TEST:  %.2e (1 std dev %.2e)' % (stats.mean(test_times), stats.stdev(test_times)))


if __name__ == '__main__':
  main()



