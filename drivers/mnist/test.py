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
import torch
import torchbraid
import torchbraid.utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import statistics as stats

import numpy as np
import matplotlib.pyplot as pyplot

from torchvision import datasets, transforms

from timeit import default_timer as timer

from mpi4py import MPI

def root_print(rank,s):
  if rank==0:
    print(s)
    sys.stdout.flush()

class StepLayer(nn.Module):
  def __init__(self,channels):
    super(StepLayer, self).__init__()
    ker_width = 3
    self.conv1 = nn.Conv2d(channels,channels,ker_width,padding=1)
    self.conv2 = nn.Conv2d(channels,channels,ker_width,padding=1)

  def forward(self, x):
    return F.relu(self.conv2(F.relu(self.conv1(x))))
# end layer

def main():
  comm = MPI.COMM_WORLD
  numprocs = comm.Get_size()
  local_steps = 3
  levels = 2

  step_layer = lambda: StepLayer(channels=2)

  parallel_nn = torchbraid.LayerParallel(comm,step_layer,local_steps*numprocs,Tf=comm.Get_size()*local_steps,max_levels=levels,max_iters=1)
  parallel_nn.setPrintLevel(0)

  fwd_lower,fwd_upper = parallel_nn.fwd_app.getStepBounds()
  bwd_lower,bwd_upper = parallel_nn.bwd_app.getStepBounds()


  parallel_nn.train()

  x = torch.rand(10,2,9,9)
  root_print(comm.Get_rank(),'FORWARD')
  root_print(comm.Get_rank(),'======================')
  y = parallel_nn(x)

  print('  %d) fwd lower,upper = ' % comm.Get_rank(),fwd_lower,fwd_upper)
  sys.stdout.flush()
  comm.barrier()

  root_print(comm.Get_rank(),'\n\nBACKWARD')
  root_print(comm.Get_rank(),'======================')
  y.backward(torch.ones(y.shape))

  print('  %d) bwd lower,upper = ' % comm.Get_rank(),bwd_lower,bwd_upper)
  sys.stdout.flush()
  comm.barrier()

  #root_print(comm.Get_rank(),'\n======================')
  #for p in parallel_nn.parameters():
  #  root_print(comm.Get_rank(),p.grad)

if __name__ == '__main__':
  main()
