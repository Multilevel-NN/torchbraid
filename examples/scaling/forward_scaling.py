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

import torch
import torch.nn as nn
import torch.nn.functional as F

import time

import getopt,sys
import argparse

# only print on rank==0
def root_print(rank,s):
  if rank==0:
    print(s)

class BasicBlock(nn.Module):
  def __init__(self,channels,timer_manager):
    super(BasicBlock, self).__init__()
    ker_width = 3
    self.conv1 = nn.Conv2d(channels,channels,ker_width,padding=1)
    self.conv2 = nn.Conv2d(channels,channels,ker_width,padding=1)
    self.timer_manager = timer_manager

  def __del__(self):
    pass

  def forward(self, x):
    with self.timer_manager.timer("forward"): 
      return F.relu(self.conv2(F.relu(self.conv1(x))))
# end layer

class ODEBlock(nn.Module):
  def __init__(self,layer,dt):
    super(ODEBlock, self).__init__()

    self.dt = dt
    self.layer = layer

  def forward(self, x):
    return x + self.dt*self.layer(x)

def build_block_with_dim(channels,timer_manager):
  b = BasicBlock(channels,timer_manager)
  return b

def compute_levels(num_steps,min_coarse_size,cfactor): 
  from math import log, floor 
  
  # we want to find $L$ such that ( max_L min_coarse_size*cfactor**L <= num_steps)
  levels =  floor(log(num_steps/min_coarse_size,cfactor))+1 

  if levels<1:
    levels = 1
  return levels
# end compute levels

# some default input arguments
###########################################

my_rank   = 0
# some default input arguments
###########################################
max_levels      = 3
max_iters       = 1
local_num_steps = 5
channels        = 16
images          = 10
image_size      = 256
Tf              = 2.0
run_serial      = False
print_level     = 0
nrelax          = 1
cfactor         = 2

# parse the input arguments
###########################################

parser = argparse.ArgumentParser()
parser.add_argument("steps",type=int,help="total numbere of steps, must be product of proc count")
parser.add_argument("--levels",    type=int,  default=max_levels,   help="maximum number of Layer-Parallel levels")
parser.add_argument("--iters",     type=int,   default=max_iters,   help="maximum number of Layer-Parallel iterations")
parser.add_argument("--channels",  type=int,   default=channels,    help="number of convolutional channels")
parser.add_argument("--images",    type=int,   default=images,      help="number of images")
parser.add_argument("--pxwidth",   type=int,   default=image_size,  help="Width/height of images in pixels")
parser.add_argument("--verbosity", type=int,   default=print_level, help="The verbosity level, 0 - little, 3 - lots")
parser.add_argument("--cfactor",   type=int,   default=cfactor,     help="The coarsening factor")
parser.add_argument("--nrelax",    type=int,   default=nrelax,      help="The number of relaxation sweeps")
parser.add_argument("--tf",        type=float, default=Tf,          help="final time for ODE")
parser.add_argument("--serial",  default=run_serial, action="store_true", help="Run the serial version (1 processor only)")
parser.add_argument("--optstr",  default=False,      action="store_true", help="Output the options string")
args = parser.parse_args()
   
# determine the number of steps
num_steps       = args.steps

# user wants us to determine how many levels
if args.levels==-1:
  min_coarse_size = 4
  args.levels = compute_levels(num_steps,min_coarse_size,args.cfactor)
# end args.levels

if args.levels:    max_levels  = args.levels
if args.iters:     max_iters   = args.iters
if args.channels:  channels    = args.channels
if args.images:    images      = args.images
if args.pxwidth:   image_size  = args.pxwidth
if args.verbosity: print_level = args.verbosity
if args.cfactor:   cfactor     = args.cfactor
if args.nrelax :   nrelax      = args.nrelax
if args.tf:        Tf          = args.tf
if args.serial:    run_serial  = args.serial


class Options:
  def __init__(self):
    self.num_steps   = args.steps
    self.max_levels  = args.levels
    self.max_iters   = args.iters
    self.channels    = args.channels
    self.images      = args.images
    self.image_size  = args.pxwidth
    self.print_level = args.verbosity
    self.cfactor     = args.cfactor
    self.nrelax      = args.nrelax
    self.Tf          = args.tf
    self.run_serial  = args.serial

  def __str__(self):
    s_net = 'net:ns=%04d_ch=%04d_im=%05d_is=%05d_Tf=%.2e' % (self.num_steps,
                                                             self.channels,
                                                             self.images,
                                                             self.image_size,
                                                             self.Tf)
    s_alg = '__alg:ml=%02d_mi=%02d_cf=%01d_nr=%02d' % (self.max_levels,
                                                       self.max_iters,
                                                       self.cfactor,
                                                       self.nrelax)
    return s_net+s_alg

opts_obj = Options()

if args.optstr==True:
  if my_rank==0:
    print(opts_obj)
  sys.exit(0)

import torchbraid
from mpi4py import MPI

comm = MPI.COMM_WORLD
my_rank   = comm.Get_rank()
last_rank = comm.Get_size()-1
numprocs  = comm.Get_size()

local_num_steps = int(num_steps/comm.Get_size())


# the number of steps is not valid, then return
if not args.steps % comm.Get_size()==0:
  if my_rank==0:
    print('error in <steps> argument, must be a multiple of proc count: %d' % comm.Get_size())
    parser.print_help()
  sys.exit(0)
# end if not args.steps

if args.serial==True and comm.Get_size()!=1:
  if my_rank==0:
    print('The <--serial> optional argument, can only be run in serial (proc count: %d)' % comm.Get_size())
    parser.print_help()
  sys.exit(0)
# end if not args.steps

print(opts_obj)

# build the neural network
###########################################

# define the neural network parameters
timer_manager = torchbraid.utils.ContextTimerManager()
basic_block = lambda: build_block_with_dim(channels,timer_manager)

# build parallel information
dt        = Tf/num_steps

# do forward propagation (in parallel)
x = torch.randn(images,channels,image_size,image_size) 

root_print(my_rank,'Number of steps: %d' % num_steps)
if run_serial:
  root_print(my_rank,'Running PyTorch: %d' % comm.Get_size())
  layers = [basic_block() for i in range(num_steps)]
  serial_nn = torch.nn.Sequential(*layers)
  with torch.no_grad(): 
    t0_parallel = time.time()
    y_serial = serial_nn(x)
    tf_parallel = time.time()

  timer_str = timer_manager.getResultString()
  print(timer_str)
else:
  root_print(my_rank,'Running TorchBraid: %d' % comm.Get_size())
  # build the parallel neural network
  parallel_nn   = torchbraid.LayerParallel(comm,basic_block,local_num_steps*numprocs,Tf,max_levels=max_levels,max_iters=max_iters)
  parallel_nn.setPrintLevel(print_level)
  parallel_nn.setSkipDowncycle(True)
  parallel_nn.setCFactor(cfactor)
  parallel_nn.setNumRelax(nrelax)
  #parallel_nn.setNumRelax(nrelax,level=0)

  t0_parallel = time.time()
  y_parallel = parallel_nn(x)
  comm.barrier()
  tf_parallel = time.time()
  comm.barrier()

  timer_str = parallel_nn.getTimersString()
  if my_rank==0:
    print(timer_str)

  timer_str = timer_manager.getResultString()
  if my_rank==0:
    print(timer_str)

  # check serial case
  serial_nn = parallel_nn.buildSequentialOnRoot()
  y_parallel = parallel_nn.getFinalOnRoot()
  if my_rank==0:
    with torch.no_grad():
      y_serial = serial_nn(x)
    
    print('error = ',torch.norm(y_serial-y_parallel)/torch.norm(y_serial))
# end if not run_serial


root_print(my_rank,'Run    Time: %.6e' % (tf_parallel-t0_parallel))
