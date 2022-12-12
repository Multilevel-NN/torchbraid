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
# Example run strings for comparing TB+NI vs. TB+MG/Opt vs. TB+MG/Opt+Local
###
#
#  +++ TB+NI (no multilevel MGRIT) +++
# $ python3 main_mgopt.py --steps 8 --samp-ratio 0.1 --lp-fwd-cfactor 2 --lp-bwd-cfactor 2  --ni-levels 2 --lp-fwd-levels 1 --lp-bwd-levels 1 --mgopt-iter 0
#    ....
#    ....
#    Train Epoch: 2 [0/5000 (0%)]        Loss: 0.080145	Time Per Batch 0.074357
#    Train Epoch: 2 [500/5000 (10%)]     Loss: 0.166830	Time Per Batch 0.073399
#
#
# +++ TB+MG/Opt (Takes the above NI solver and adds 1 epoch of MG/Opt)  (no multilevel MGRIT) +++
# $ python3 main_mgopt.py --steps 12 --samp-ratio 0.1 --epochs 2 --mgopt-printlevel 1 --ni-levels 2 --mgopt-levels 2 --mgopt-nrelax-pre 2 --mgopt-nrelax-post 2 --lp-fwd-cfactor 2 --lp-bwd-cfactor 2 --lp-fwd-levels 1 --lp-bwd-levels 1 --lp-iters 1
#    ....
#    ....
#    Train Epoch: 1 [50/5000 (1%)]     Loss: 0.023389	Time Per Batch 0.780689
#    Train Epoch: 1 [550/5000 (11%)]     Loss: 0.037635	Time Per Batch 0.781935
#
#
# ++++  TB+MG/Opt+Local (Takes the above NI solver and adds 2 epoch of MG/Opt with purely local relaxation on each level) +++
# $ python3 main_mgopt.py --steps 8 --samp-ratio 0.1 --epochs 2 --mgopt-printlevel 1 --ni-levels 2 --mgopt-levels 2 --mgopt-nrelax-pre 2 --mgopt-nrelax-post 2 --lp-fwd-cfactor 2 --lp-bwd-cfactor 2 --lp-fwd-levels 1 --lp-bwd-levels 1 --lp-fwd-finefcf --lp-bwd-finefcf --lp-fwd-relaxonlycg --lp-bwd-relaxonlycg --lp-fwd-finalrelax --lp-iters 1
#
#
#   No Forward relax only
# $ python3 main_mgopt.py --steps 12 --samp-ratio 0.1 --epochs 4 --mgopt-printlevel 1 --ni-levels 2 --mgopt-levels 2 --mgopt-nrelax-pre 2 --mgopt-nrelax-post 2 --lp-fwd-cfactor 2 --lp-bwd-cfactor 2 --lp-fwd-levels 1 --lp-bwd-levels 1 --lp-bwd-finefcf --lp-bwd-relaxonlycg --lp-iters 1
#    ....
#    ....
#   Train Epoch: 1 [50/5000 (1%)]     Loss: 0.026627	Time Per Batch 1.354365
#   Train Epoch: 1 [550/5000 (11%)]     Loss: 0.038312	Time Per Batch 1.314453
#
#
# Same as above, but turned off relaxonlycg 
# $ python3 main_mgopt.py --steps 12 --samp-ratio 0.1 --epochs 4 --mgopt-printlevel 1 --ni-levels 2 --mgopt-levels 2 --mgopt-nrelax-pre 2 --mgopt-nrelax-post 2 --lp-fwd-cfactor 2 --lp-bwd-cfactor 2 --lp-fwd-levels 1 --lp-bwd-levels 1  --lp-iters 1
#    ....
#    ....
#   Train Epoch: 1 [50/5000 (1%)]     Loss: 0.023389	Time Per Batch 0.791520
#   Train Epoch: 1 [550/5000 (11%)]     Loss: 0.037635	Time Per Batch 0.786753
#
#
# +++ relax_onlycg used both ways +++
#  $python3 main_mgopt.py --steps 12 --samp-ratio 0.1 --epochs 4 --mgopt-printlevel 1 --ni-levels 2 --mgopt-levels 2 --mgopt-nrelax-pre 2 --mgopt-nrelax-post 2 --lp-fwd-cfactor 2 --lp-bwd-cfactor 2 --lp-fwd-levels 1 --lp-bwd-levels 1 --lp-bwd-finefcf --lp-bwd-relaxonlycg --lp-iters 1 --lp-fwd-relaxonlycg --lp-fwd-finefcf 
#
#   Train Epoch: 1 [50/5000 (1%)]     Loss: 1.007567	Time Per Batch 1.445430
#   Train Epoch: 1 [550/5000 (11%)]     Loss: 1.922850	Time Per Batch 1.413595


from __future__ import print_function
import numpy as np

import torch
from torchvision import datasets, transforms
from mpi4py import MPI
from utils import parse_args, ParallelNet
from torchbraid.mgopt import mgopt_solver, root_print
from torchbraid.utils import getDevice
import torchbraid

def main():
  
  ##
  # Parse command line args (function defined above)
  args = parse_args()
  procs = MPI.COMM_WORLD.Get_size()
  rank  = MPI.COMM_WORLD.Get_rank()
  root_print(rank, 1, 1, 'TORCHBRAID REV: %s\n' % torchbraid.utils.git_rev())

  my_device,my_host = getDevice(MPI.COMM_WORLD)
  
  #torch.set_num_threads(6)

  ##
  # Load training and testing data, while reducing the number of samples (if desired) for faster execution
  transform = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))
                                 ])
  if args.use_fashion:
    if rank==0:
      print('Using fashion MNIST...')
    dataset = datasets.FashionMNIST('./fashion-data', download=False,transform=transform)
  else:
    if rank==0:
      print('Using digit MNIST...')
    dataset = datasets.MNIST('./digit-data', download=False,transform=transform)
  train_size = int(50000*args.samp_ratio)
  test_size = int(10000*args.samp_ratio)
  #
  if (train_size % args.batch_size) or (test_size % args.batch_size) != 0:
    print("\nSize of train and test datasets must divide evenly with batch size")
    return
  #
  train_set = torch.utils.data.Subset(dataset,range(train_size))
  test_set  = torch.utils.data.Subset(dataset,range(train_size,train_size+test_size))
  #
  train_loader = torch.utils.data.DataLoader(train_set,batch_size=args.batch_size,shuffle=True)
  test_loader = torch.utils.data.DataLoader(test_set,batch_size=args.batch_size,shuffle=True)
  if rank==0:
    print("\nTraining setup:  Batch size:  " + str(args.batch_size) + "  Sample ratio:  " + str(args.samp_ratio) + "  MG/Opt Epochs:  " + str(args.epochs) )
  
  ##
  # Compute number of nested iteration steps, going from fine to coarse
  ni_steps = np.array([int(args.steps/(args.ni_rfactor**(args.ni_levels-i-1))) for i in range(args.ni_levels)])
  ni_steps = ni_steps[ ni_steps != 0 ]
  if rank==0:
    print("\nNested iteration steps:  " + str(ni_steps))

  ##
  # Define ParNet parameters for each nested iteration level, starting from fine to coarse (flip ni_steps)
  networks = [] 
  for nsteps in np.flip(ni_steps): 
    networks.append(('Factory', {'channels'          : args.channels, 
                                 'global_steps'      : nsteps,
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
  #optims = [ ("pytorch_sgd", { 'lr':args.lr, 'momentum':0.9}) for i in range(len(ni_steps)) ]
  optims = [ ("pytorch_adam", { 'lr':0.001, 'betas':(0.9, 0.999), 'eps':1e-08 }) for i in range(len(ni_steps)) ]
  
  ##
  # The preserve optim command line argument controls the mgopt phase below.  Here, we have a manual flag
  preserve_optim = True

  ##
  # Initialize MG/Opt solver with nested iteration 
  interp_params = "tb_parallel_get_injection_interp_params"
  #interp_params = "tb_get_injection_interp_params"
  epochs = args.NIepochs
  mgopt_printlevel = args.mgopt_printlevel
  log_interval = args.log_interval
  mgopt = mgopt_solver(device=my_device)

  model_factory = lambda level,**kwargs: ParallelNet(**kwargs)

  mgopt.initialize_with_nested_iteration(model_factory,ni_steps, train_loader, test_loader,
          networks, interp_params=interp_params, epochs=epochs, log_interval=log_interval,
          mgopt_printlevel=mgopt_printlevel, optims=optims, seed=args.seed, 
          preserve_optim=preserve_optim, zero_init_guess=args.zero_init_guess) 
   
  mgopt.options_used()
  
  ##
  # Turn on for fixed-point test.  
  # Works when running  $$ python3 main_mgopt.py --samp-ratio 0.002 --lp-fwd-cfactor 2 --lp-bwd-cfactor 2 --mgopt-printlevel 3 --batch-size 1
  if False:
    import torch.nn as nn
    criterion = nn.CrossEntropyLoss()
    train_set = torch.utils.data.Subset(dataset, [1])
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=1,shuffle=False)
    for (data,target) in train_loader:  pass
    model = mgopt.levels[0].model
    with torch.no_grad():
      model.eval()
      output = model(data)
      loss = model.compose(criterion, output, target)
    
    if rank==0:
      print("Doing fixed point test.  Loss on single training example should be zero: " + str(loss.item()))
    model.train()

 # Can change MGRIT options from NI to MG/Opt with the following
 #mgopt.levels[0].model.parallel_nn.setFwdNumRelax(0,level=0) 
  
  ##
  # Run the MG/Opt solver
  #   Note: that we use the default restrict and interp options, but these can be modified on a per-level basis
  if( args.mgopt_iter > 0):
    epochs = args.epochs
    preserve_optim = args.preserve_optim
    #line_search = ('tb_adam_no_ls', {'ls_params' : {}})
    line_search = ('tb_simple_backtrack_ls', {'ls_params' : {'n_line_search':6, 'alpha':1.0, 'c1':0.0001}} )
    #line_search = ('tb_simple_ls', {'ls_params' : {'alphas' : [0.01, 0.1, 0.5, 1.0, 2.0, 4.0]}} )
    #line_search = ('tb_simple_weighting', {'ls_params' : {'alpha' : 0.001}} )
    restrict_params = "tb_parallel_get_injection_restrict_params"
    #restrict_params = "tb_get_injection_restrict_params"
    restrict_grads = "tb_parallel_get_injection_restrict_params"
    #restrict_grads = "tb_get_injection_restrict_params"
    log_interval = args.log_interval
    mgopt_printlevel = args.mgopt_printlevel
    mgopt_iter = args.mgopt_iter
    mgopt_levels = args.mgopt_levels
    mgopt_tol=0
    nrelax_pre = args.mgopt_nrelax_pre
    nrelax_post = args.mgopt_nrelax_post
    nrelax_coarse = args.mgopt_nrelax_coarse
    print("MGOPT SOLVE STARTS")
    mgopt.mgopt_solve(train_loader, test_loader, epochs=epochs,
            log_interval=log_interval, mgopt_tol=mgopt_tol,
            mgopt_iter=mgopt_iter, nrelax_pre=nrelax_pre,
            nrelax_post=nrelax_post, nrelax_coarse=nrelax_coarse,
            mgopt_printlevel=mgopt_printlevel, mgopt_levels=mgopt_levels,
            line_search=line_search, restrict_params=restrict_params, 
            restrict_grads=restrict_grads, preserve_optim=preserve_optim)
   
    print(mgopt)
    mgopt.options_used()
  ##
  


if __name__ == '__main__':
  main()



