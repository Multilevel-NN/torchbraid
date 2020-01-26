import torch
import torch.nn as nn
import torch.nn.functional as F

import torchbraid
import time

import getopt,sys

from mpi4py import MPI

class BasicBlock(nn.Module):
  def __init__(self,channels):
    super(BasicBlock, self).__init__()
    ker_width = 3
    self.conv1 = nn.Conv2d(channels,channels,ker_width,padding=1)
    self.conv2 = nn.Conv2d(channels,channels,ker_width,padding=1)

  def __del__(self):
    pass

  def forward(self, x):
    return F.relu(self.conv2(F.relu(self.conv1(x))))
# end layer

class ODEBlock(nn.Module):
  def __init__(self,layer,dt):
    super(ODEBlock, self).__init__()

    self.dt = dt
    self.layer = layer

  def forward(self, x):
    return x + self.dt*self.layer(x)

def build_block_with_dim(channels):
  b = BasicBlock(channels)
  return b

# some input arguments
max_levels      = 3
max_iters       = 1
local_num_steps = 5
channels        = 16
images          = 10
image_size      = 256

# define the neural network parameters
Tf = 2.0
basic_block = lambda: build_block_with_dim(channels)

# build parallel information
comm = MPI.COMM_WORLD
my_rank   = comm.Get_rank()
last_rank = comm.Get_size()-1
num_steps = local_num_steps*comm.Get_size()
dt        = Tf/num_steps

# build the parallel neural network
parallel_nn   = torchbraid.Model(comm,basic_block,local_num_steps,Tf,max_levels=max_levels,max_iters=max_iters)

# do forward propagation (in parallel)
x = torch.randn(images,channels,image_size,image_size) 

t0_parallel = time.time()
y_parallel = parallel_nn(x)
comm.barrier()
tf_parallel = time.time()

comm.barrier()

## communicate forward prop answer form Layer-Parallel
if last_rank!=0:
  if my_rank==0:
    y_parallel = comm.recv(source=last_rank,tag=12)
  elif my_rank==last_rank:
    comm.send(y_parallel,dest=0,tag=12)
# end if last_rank

if True:
  # build up the serial neural network
  #ode_layers    = [ODEBlock(l,dt) for l in parallel_nn.local_layers.children()]
  #remote_layers = comm.Get_size()*[None]
  #remote_layers = ode_layers
  
  #if last_rank>0:
  #	if my_rank==0:
  #		for i in range(1,comm.Get_size()):
  #			remote_layers += comm.recv(source=i,tag=12)
  #	else:
  #		comm.send(ode_layers,dest=0,tag=12)
  #serial_nn = torch.nn.Sequential(*remote_layers)
  
  # check error on root node
  if my_rank==0:
    remote_layers = [ODEBlock(basic_block(),dt) for l in range(num_steps)]
    serial_nn = torch.nn.Sequential(*remote_layers)
  
    with torch.no_grad(): 
    	t0_serial = time.time()
    	y_serial = serial_nn(x)
    	tf_serial = time.time()
    
    val = torch.norm(y_parallel-y_serial).item()
    print('error = %.6e' % val)
    
    print('Serial Time     = %.4e' % (tf_serial-t0_serial))
    print('Parallel Time   = %.4e' % (tf_parallel-t0_parallel))
    print('Serial/Parallel = %.4e' % ((tf_serial-t0_serial)/(tf_parallel-t0_parallel)))
    print(' ------- end ------- ')
  # end error check
