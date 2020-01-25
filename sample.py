import torch
import torch.nn as nn
import torch.nn.functional as F

import torchbraid

from mpi4py import MPI

class BasicBlock(nn.Module):
  def __init__(self,dim):
    super(BasicBlock, self).__init__()
    self.lin = nn.Linear(dim, dim)
    self.weight = self.lin.weight
    self.bias = self.lin.bias

  def __del__(self):
    pass

  def forward(self, x):
    return F.relu(self.lin(x))
# end layer

class ODEBlock(nn.Module):
  def __init__(self,layer,dt):
    super(ODEBlock, self).__init__()

    self.dt = dt
    self.layer = layer

  def __del__(self):
    pass

  def forward(self, x):
    return x + self.dt*self.layer(x)

def build_block_with_dim(dim):
  b = BasicBlock(dim)
  #nn.init.constant_(b.weight,2.0)
  #nn.init.constant_(b.bias,0.0)
  return b

# define the neural network parameters
dim = 20
Tf = 2.0
local_num_steps = 10
basic_block = lambda: build_block_with_dim(dim)

# build parallel information
comm = MPI.COMM_WORLD
my_rank   = comm.Get_rank()
last_rank = comm.Get_size()-1
num_steps = local_num_steps*comm.Get_size()
dt        = Tf/num_steps

# build the parallel neural network
parallel_nn   = torchbraid.Model(comm,basic_block,local_num_steps,Tf,max_levels=4,max_iters=10)

ode_layers    = [ODEBlock(l,dt) for l in parallel_nn.local_layers.children()]

# build up the serial neural network
ode_layers    = [ODEBlock(l,dt) for l in parallel_nn.local_layers.children()]
remote_layers = comm.Get_size()*[None]
remote_layers = ode_layers

if last_rank>0:
	if my_rank==0:
		for i in range(1,comm.Get_size()):
			remote_layers += comm.recv(source=i,tag=12)
	else:
		comm.send(ode_layers,dest=0,tag=12)
serial_nn = torch.nn.Sequential(*remote_layers)

# do forward propagation (in parallel)
x = torch.randn(5,dim) 
y_parallel = parallel_nn(x)

comm.barrier()

# communicate forward prop answer form Layer-Parallel
if last_rank!=0:
  if my_rank==0:
    y_parallel = comm.recv(source=last_rank,tag=12)
  elif my_rank==last_rank:
    comm.send(y_parallel,dest=0,tag=12)


# check error on root node
if my_rank==0:

  with torch.no_grad(): 
    y_serial = serial_nn(x)

  val = torch.norm(y_parallel-y_serial).item()
  print('error = %.6e' % val)

  print(' ------- end ------- ')
# end error check
