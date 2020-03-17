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
    xm = torch.randn(5,dim) 
    xm.requires_grad = True

    xf = xm.detach().clone()
    xf.requires_grad = True
    
    yf = f(xf)
    yf.backward(torch.ones(xf.shape))
    print('xf grad')
    print(xf.grad)

    ym = m(xm)
    ym.backward(torch.ones(xm.shape))
    print('xm grad')
    print(xm.grad)

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
