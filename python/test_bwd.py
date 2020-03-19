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

class LinearBlock(nn.Module):
  def __init__(self,dim=10):
    super(LinearBlock, self).__init__()

  def forward(self, x):
    return x
# end layer

class NonlinearBlock(nn.Module):
  def __init__(self,dim=10):
    super(NonlinearBlock, self).__init__()

  def forward(self, x):
    return 0.5*x**2
# end layer

class ReLUBlock(nn.Module):
  def __init__(self,dim=10):
    super(ReLUBlock, self).__init__()
    self.lin = nn.Linear(dim, dim,bias=True)

    w = torch.randn(dim,dim)
    self.lin.weight = torch.nn.Parameter(w)

    b = torch.randn(dim)
    self.lin.bias = torch.nn.Parameter(b)

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
  def test_linearNetSerial(self):
    dim = 2
    basic_block = lambda: LinearBlock(dim)

    x0 = torch.randn(5,dim) # forward initial cond
    w0 = torch.randn(5,dim) # adjoint initial cond
    self.backForwardProp(dim,basic_block,x0,w0)

    print('----------------------------')
  # end test_linearNet

  def test_nonlinearNetSerial(self):
    dim = 2
    basic_block = lambda: NonlinearBlock(dim)

    x0 = torch.randn(5,dim) # forward initial cond
    w0 = torch.randn(5,dim) # adjoint initial cond
    self.backForwardProp(dim,basic_block,x0,w0)

    print('----------------------------')
  # end test_linearNet

  def test_reLUNetSerial(self):
    dim = 2
    basic_block = lambda: ReLUBlock(dim)

    x0 = torch.randn(5,dim) # forward initial cond
    w0 = torch.randn(5,dim) # adjoint initial cond
    self.backForwardProp(dim,basic_block,x0,w0)

    print('----------------------------')
  # end test_linearNet

  def atest_gradient(self):
    dim = 2
    basic_block = lambda: ReLUBlock(dim)

    x0 = torch.randn(5,dim) # forward initial cond
    w0 = torch.randn(5,dim) # adjoint initial cond

    Tf = 2.0
    num_steps = 10

    # run forward/backward propgation

    # this is the torchbraid class being tested 
    #######################################
    m = torchbraid.Model(MPI.COMM_WORLD,basic_block,num_steps,Tf,max_levels=1,max_iters=1)
    m.setPrintLevel(0)

    # this is the reference torch "solution"
    #######################################
    dt = Tf/num_steps
    ode_layers = [ODEBlock(l,dt) for l in m.local_layers.children()]
    f = torch.nn.Sequential(*ode_layers)

    # propogation with torch
    #######################################
    xf = x0.clone()
    xf.requires_grad = True
    
    f.zero_grad()
    for p in f.parameters():
      print(p.grad)

    wf = f(xf)
    wf.backward(w0)

    print('\ntorch grad parameters')
    for p in f.parameters():
      print(p.grad)

    # propogation with torchbraid
    #######################################
    xm = x0.clone()
    xm.requires_grad = True

    m.zero_grad()
    for p in m.parameters():
      print(p.grad)

    wm = m(xm)
    wm.backward(w0)

    print('\ntorchbraid grad parameters')
    for p in m.parameters():
      print(p.grad)

  def backForwardProp(self,dim, basic_block,x0,w0,max_levels=1,max_iters=1):
    Tf = 2.0
    num_steps = 10

    # this is the torchbraid class being tested 
    #######################################
    m = torchbraid.Model(MPI.COMM_WORLD,basic_block,num_steps,Tf,max_levels=max_levels,max_iters=max_iters)
    m.setPrintLevel(0)

    # this is the reference torch "solution"
    #######################################
    dt = Tf/num_steps
    ode_layers = [ODEBlock(l,dt) for l in m.local_layers.children()]
    f = torch.nn.Sequential(*ode_layers)

    # run forward/backward propgation

    # propogation with torch
    #######################################
    xf = x0.clone()
    xf.requires_grad = True
    
    wf = f(xf)
    wf.backward(w0)

    # propogation with torchbraid
    #######################################
    xm = x0.clone()
    xm.requires_grad = True

    wm = m(xm)
    wm.backward(w0)

    # compare the solutions
    #######################################

    # check some values
    if m.getMPIData().getRank()==m.getMPIData().getSize()-1:
      self.assertTrue(torch.norm(wm)>0.0)
      self.assertTrue(torch.norm(wf)>0.0)

      print('\n')
      print('fwd error = %.6e' % (torch.norm(wm-wf)/torch.norm(wf)))
      print('grad error = %.6e' % (torch.norm(xm.grad-xf.grad)/torch.norm(xf.grad)))

      self.assertTrue(torch.norm(wm-wf)/torch.norm(wf)<=1e-16)
      self.assertTrue((torch.norm(xm.grad-xf.grad)/torch.norm(xf.grad))<=1e-16)

      for pf,pm in zip(f.parameters(),m.parameters()):
        print('p grad error = %.6e' % (torch.norm(pf.grad-pm.grad)), pf.grad.shape)
        self.assertTrue(torch.norm(pf.grad-pm.grad)<=1e-16)
        
  # test_forwardPropSerial

if __name__ == '__main__':
  unittest.main()
