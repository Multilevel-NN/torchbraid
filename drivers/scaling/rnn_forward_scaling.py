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

import torchvision
import torchvision.transforms as transforms

import torchbraid
import time

import getopt,sys
import argparse

from mpi4py import MPI

# only print on rank==0
def root_print(rank,s):
  if rank==0:
    print(s)


# LSTM tutorial: https://pytorch.org/docs/stable/nn.html

class RNN_BasicBlock(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers):
    super(RNN_BasicBlock, self).__init__()

    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

  def __del__(self):
    pass

  def forward(self, x):
    # Set initial hidden and cell states

    # Inputs: input, (h_0, c_0)
    # input of shape (batch, seq_len, input_size) or (seq_len, batch, input_size)
    # h_0 of shape (num_layers * num_directions, batch, hidden_size)
    # c_0 of shape (num_layers * num_directions, batch, hidden_size)

    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
    c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
    # Forward propagate LSTM
    output, (hn, cn) = self.lstm(x, (h0, c0))

    # Outputs: output, (h_n, c_n)
    # output of shape (batch, seq_len, num_directions * hidden_size) or (seq_len, batch, num_directions * hidden_size)
    # h_n of shape (num_layers * num_directions, batch, hidden_size)
    # c_n of shape (num_layers * num_directions, batch, hidden_size)

    return output

  # From the second block to the last block in the sequence
  ###########################################
  # def forward(self, x, hn, cn):
  #   h0 = hn
  #   c0 = cn
  #   output, (hn, cn) = self.lstm(x, (h0, c0))
  #   return output



class ODEBlock(nn.Module):
  def __init__(self,layer,dt):
    super(ODEBlock, self).__init__()

    self.dt = dt
    self.layer = layer

  def forward(self, x):
    return x + self.dt*self.layer(x)

def RNN_build_block_with_dim(input_size, hidden_size, num_layers):
  b = RNN_BasicBlock(input_size, hidden_size, num_layers) # channels = hidden_size
  return b

# some default input arguments
###########################################

comm = MPI.COMM_WORLD
my_rank   = comm.Get_rank()
last_rank = comm.Get_size()-1

# some default input arguments
###########################################
max_levels      = 3
max_iters       = 1
local_num_steps = 5
num_steps       = int(local_num_steps*comm.Get_size())
# For CNN
###########################################
# channels        = 16
# images          = 10
# image_size      = 256

# For RNN
###########################################
channels        = 1
images          = 10
image_size      = 28

Tf              = 2.0
run_serial      = False
print_level     = 0
nrelax          = 1
cfactor         = 2

# parse the input arguments
###########################################

parser = argparse.ArgumentParser()
parser.add_argument("steps",type=int,help="total number of steps, must be product of proc count (p=%d)" % comm.Get_size())
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
   
# determine the number of steps
num_steps       = args.steps
local_num_steps = int(num_steps/comm.Get_size())

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
    self.num_procs   = comm.Get_size()
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
  if comm.Get_rank()==0:
    print(opts_obj)
  sys.exit(0)
    
print(opts_obj)


# set hyper-parameters
###########################################
sequence_length = 28 # total number of time steps for each sequence
input_size = 28 # input size for each time step in a sequence
hidden_size = 64
num_layers = 2


# build the neural network
###########################################

# define the neural network parameters
basic_block = lambda: RNN_build_block_with_dim(input_size, hidden_size, num_layers)

# build parallel information
dt        = Tf/num_steps

# do forward propagation (in parallel)
###########################################
# generate randomly initialized data
###########################################
x = torch.randn(images,channels,sequence_length,input_size)

root_print(my_rank,'Number of steps: %d' % num_steps)

if run_serial:

  root_print(my_rank,'Running PyTorch: %d' % comm.Get_size())

  ###########################################
  # Unlike CNN's basic_block, RNN's basic_block includes the LSTM network with layers across a certain number of time steps.
  # For RNN, basic_block has to be the block of the number of time steps over a sequence, e.g., Each block includes LSTM networks along with two time steps.

  # Then, how to break the time steps in a sequence into num_steps?
  # We need (num_steps) basic blocks, and each block contains (sequence length / num_steps) time steps.

  # For RNN, [for loop] is no longer needed anymore
  ###########################################
  block = basic_block()

  serial_nn = block

  # How to connect the separated sub-sequences/blocks?
  # Compare the results between final_output1 and final_output2
  # result = serial_nn(X_sub0)
  # final_output1 = serial_nn(X_sub1,result)
  # final_output2 = serial_nn(X_sub0,X_sub1)


  with torch.no_grad():
    t0_parallel = time.time()

    for i in range(len(x)):
      images = x[i].reshape(-1, sequence_length, input_size) # (batch_size, seq_len, features)
      y_serial = serial_nn(images)
    
    tf_parallel = time.time()

"""
# TODO: distribute Neural Network
###########################################
on root
  bb = build basic block
  MPI_Send (bb, to all procs)
else
  bb = MPI_Recv (bb, root)

# TODO: distribute data
###########################################
on root
  MPI_Send (data, to all procs)
else
  MPI_Recv (data)
"""

# max_levels = 1 means serial version

"""
else:
  root_print(my_rank,'Running TorchBraid: %d' % comm.Get_size())
  # build the parallel neural network
  parallel_nn   = torchbraid.Model(comm,basic_block,local_num_steps,Tf,max_levels=max_levels,max_iters=max_iters)
  parallel_nn.setPrintLevel(print_level)
  parallel_nn.setCFactor(cfactor)
  parallel_nn.setNumRelax(nrelax)

  t0_parallel = time.time()
  y_parallel = parallel_nn(x)
  comm.barrier()
  tf_parallel = time.time()
  comm.barrier()

  # check serial case
  serial_nn = parallel_nn.buildSequentialOnRoot()
  y_parallel = parallel_nn.getFinalOnRoot()
  if my_rank==0:
    with torch.no_grad():
      y_serial = serial_nn(x)
    
    print('error = ',torch.norm(y_serial-y_parallel)/torch.norm(y_serial))
# end if not run_serial
"""

root_print(my_rank,'Run    Time: %.6e' % (tf_parallel-t0_parallel))