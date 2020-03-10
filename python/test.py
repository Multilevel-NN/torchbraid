import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest
import sys
import numpy as np

import torchbraid

import faulthandler
faulthandler.enable()

from mpi4py import MPI

class BasicBlock(nn.Module):
  def __init__(self,dim=10):
    super(BasicBlock, self).__init__()
    self.lin = nn.Linear(dim, dim)

  def forward(self, x):
    return F.relu(self.lin(x))
# end layer

class ODEBlock(nn.Module):
  def __init__(self,layer,dt):
    super(ODEBlock, self).__init__()

    self.dt = dt
    self.layer = layer
  def forward(self, x):
    return x + self.dt*self.layer(x)

class TestTorchBraid(unittest.TestCase):

  def test_cloneInitVector(self):
    x         = torch.randn(5,10) 
    Tf        = 2.0
    num_steps = 1
  
    m = torchbraid.Model(MPI.COMM_WORLD,BasicBlock,num_steps,Tf)
    m.setInitial(x)
  
    out = torchbraid.cloneInitVector(m)
    nrm = torch.norm(x-out)
    torchbraid.freeVector(m,out)
    self.assertEqual(nrm,0.0)
  # end test_cloneInitVector

  def test_cloneVector(self):
    x         = torch.randn(5,10) 
    Tf        = 2.0
    num_steps = 1
  
    m = torchbraid.Model(MPI.COMM_WORLD,BasicBlock,num_steps,Tf)
  
    out = torchbraid.cloneVector(m,x)
    nrm = torch.norm(x-out)
    torchbraid.freeVector(m,out)
    self.assertEqual(nrm,0.0)
  # end test_cloneInitVector

  def test_addVector(self):
    Tf        = 2.0
    num_steps = 1
  
    m = torchbraid.Model(MPI.COMM_WORLD,BasicBlock,num_steps,Tf)

    x     = 2.0*torch.ones(2,3) 
    y     = 1.0*torch.ones(2,3) 
    alpha =  8.0
    beta  =  2.0
    z = alpha*x+beta*y
        # 8.0*2 + 2.0*1 = 18.0
  
    torchbraid.addVector(m,alpha,x,beta,y)
    nrm = torch.norm(z-y)
    self.assertEqual(nrm,0.0)
  # end test_addVector

  def test_vectorNorm(self):
    Tf        = 2.0
    num_steps = 1
    x         = torch.randn(5,10) 
    norm_x    = torch.norm(x) 

    m = torchbraid.Model(MPI.COMM_WORLD,BasicBlock,num_steps,Tf)
  
    norm_x_c = torchbraid.vectorNorm(m,x)
    self.assertEqual(norm_x,norm_x_c)
  # end test_addVector

  def test_bufSize(self):
    Tf         = 2.0
    num_steps  = 1
    x          = torch.randn(5,10,3,9) 
    cnt        = 5*10*3*9
    sizeof_dbl = np.dtype(np.float).itemsize
    sizeof_int = np.dtype(np.int32).itemsize

    m = torchbraid.Model(MPI.COMM_WORLD,BasicBlock,num_steps,Tf)
    m.setInitial(x)
  
    cnt_c = torchbraid.bufSize(m) 
    # self.assertEqual(cnt*sizeof_dbl+sizeof_dbl+sizeof_int,cnt_c)
  # end test_bufSize

#   def test_bufpackunpack(self):
#     Tf         = 2.0
#     num_steps  = 1
#     x          = torch.randn(3,2,4) 
# 
#     l = 0.0
#     for i in range(3):
#       for j in range(2):
#         for k in range(4):
#           x[i,j,k] = l
#           l += 1.0
#     # end for i,j,k
# 
#     m = torchbraid.Model(MPI.COMM_WORLD,BasicBlock,num_steps,Tf)
#     m.setInitial(x)
# 
#     # we allocate the buffer
#     buffer = torchbraid.allocBuffer(m) # this is allocated from x0
# 
#     # we pack the solution into the buffer
#     print('pack')
#     torchbraid.pack(m,x,buffer,3)
# 
#     # we unpack the solution and build a new tensor
#     print('unpack')
#     u,l = torchbraid.unpack(m,buffer)
# 
#     # we free the buffer
#     print('free')
#     torchbraid.freeBuffer(m,buffer)
# 
#     print('assert')
#     self.assertEqual(l,3)
#     self.assertEqual(torch.norm(u-x),0.0)

  def test_forwardPropSerial(self):
    dim = 10
    basic_block = lambda: BasicBlock(dim)

    Tf = 2.0
    num_steps = 10

    # this is the class being tested (the forward propagation)
    m = torchbraid.Model(MPI.COMM_WORLD,basic_block,num_steps,Tf,max_levels=2,max_iters=10)
    m.setPrintLevel(0)

    # this is the reference "solution"
    dt = Tf/num_steps
    ode_layers = [ODEBlock(l,dt) for l in m.local_layers.children()]
    f = torch.nn.Sequential(*ode_layers)

    # run forward propgation
    x = torch.randn(5,dim) 
    ym = m(x)
    yf = f(x)

    # check some values
    if m.getMPIData().getRank()==m.getMPIData().getSize()-1:
      self.assertTrue(torch.norm(ym)>0.0)
      self.assertTrue(torch.norm(yf)>0.0)
      self.assertTrue(torch.norm(ym-yf)<=1e-6)
  # test_forwardPropSerial

#   def test_coarsenRefine(self):
#     dim = 10
#     basic_block = lambda: BasicBlock(dim)
# 
#     Tf = 2.0
#     num_steps = 10
# 
#     def coarsen(x,level):
#       return x.clone()
#     def refine(x,level):
#       return x.clone()
# 
#     # this is the class being tested (the forward propagation)
#     m = torchbraid.Model(MPI.COMM_WORLD,basic_block,num_steps,Tf,max_levels=2,max_iters=10,
#                          coarsen=coarsen,
#                          refine=refine)
#     m.setPrintLevel(0)
# 
#     # this is the reference "solution"
#     dt = Tf/num_steps
#     ode_layers = [ODEBlock(l,dt) for l in m.local_layers.children()]
#     f = torch.nn.Sequential(*ode_layers)
# 
#     # run forward propgation
#     x = torch.randn(5,dim) 
#     ym = m(x)
#     yf = f(x)
# 
#     # check some values
#     if m.getMPIData().getRank()==m.getMPIData().getSize()-1:
#       self.assertTrue(torch.norm(ym)>0.0)
#       self.assertTrue(torch.norm(yf)>0.0)
#       self.assertTrue(torch.norm(ym-yf)<=1e-6)
#   # test_forwardPropSerial

if __name__ == '__main__':
  unittest.main()
