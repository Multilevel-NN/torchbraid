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

class ReLUBlock(nn.Module):
  def __init__(self,dim=10):
    super(ReLUBlock, self).__init__()
    self.lin = nn.Linear(dim, dim,bias=True)

    w = torch.randn(dim,dim)
    w = 2.3*torch.ones(dim,dim)
    self.lin.weight = torch.nn.Parameter(w)

    b = torch.randn(dim)
    b = -1.22*torch.ones(dim)
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
  def test_linearNet_Exact(self):
    dim = 2
    basic_block = lambda: LinearBlock(dim)

    x0 = torch.randn(5,dim) # forward initial cond
    w0 = torch.randn(5,dim) # adjoint initial cond
    max_levels = 1
    max_iters = 1
    self.backForwardProp(dim,basic_block,x0,w0,max_levels,max_iters)

    print('----------------------------')
  # end test_linearNet

  def test_linearNet_Approx(self):
    dim = 2
    basic_block = lambda: LinearBlock(dim)

    x0 = torch.randn(5,dim) # forward initial cond
    w0 = torch.randn(5,dim) # adjoint initial cond
    max_levels = 3
    max_iters = 8
    self.backForwardProp(dim,basic_block,x0,w0,max_levels,max_iters,test_tol=1e-6)

    print('----------------------------')
  # end test_linearNet

  def test_reLUNet_Exact(self):
    dim = 2
    basic_block = lambda: ReLUBlock(dim)

    x0 = torch.randn(5,dim) # forward initial cond
    w0 = torch.randn(5,dim) # adjoint initial cond
    x0 = 12.0*torch.ones(5,dim) # forward initial cond
    w0 = 8.0*torch.ones(5,dim) # adjoint initial cond
    max_levels = 1
    max_iters = 1
    self.backForwardProp(dim,basic_block,x0,w0,max_levels,max_iters)

    print('----------------------------')
  # end test_reLUNetSerial

  def test_reLUNet_Approx(self):
    dim = 2
    basic_block = lambda: ReLUBlock(dim)

    x0 = torch.randn(5,dim) # forward initial cond
    w0 = torch.randn(5,dim) # adjoint initial cond
    x0 = 12.0*torch.ones(5,dim) # forward initial cond
    w0 = 8.0*torch.ones(5,dim) # adjoint initial cond
    max_levels = 3
    max_iters = 8
    self.backForwardProp(dim,basic_block,x0,w0,max_levels,max_iters,test_tol=1e-6)

    print('----------------------------')
  # end test_reLUNetSerial

  def copyParameterGradToRoot(self,m):
    comm     = m.getMPIData().getComm()
    my_rank  = m.getMPIData().getRank()
    num_proc = m.getMPIData().getSize()
 
    params = [p.grad for p in list(m.parameters())]

    if len(params)==0:
      return params

    if my_rank==0:
      for i in range(1,num_proc):
        remote_p = comm.recv(source=i,tag=77)
        params.extend(remote_p)

      return params
    else:
      comm.send(params,dest=0,tag=77)
      return None
  # end copyParametersToRoot

  def backForwardProp(self,dim, basic_block,x0,w0,max_levels=1,max_iters=1,test_tol=1e-16):
    Tf = 2.0
    num_steps = 10

    # this is the torchbraid class being tested 
    #######################################
    m = torchbraid.Model(MPI.COMM_WORLD,basic_block,num_steps,Tf,max_levels=max_levels,max_iters=max_iters)
    m.setPrintLevel(1)

    w0 = m.copyVectorFromRoot(w0)

    # this is the reference torch "solution"
    #######################################
    dt = Tf/num_steps
    f = m.buildSequentialOnRoot()

    # run forward/backward propgation

    # propogation with torchbraid
    #######################################
    xm = x0.clone()
    xm.requires_grad = True

    wm = m(xm)
    wm.backward(w0)

    wm = m.getFinalOnRoot()
    m_param_grad = self.copyParameterGradToRoot(m)

    # check some values
    if m.getMPIData().getRank()==0:
 
      compute_grad = True

      # propogation with torch
      #######################################
      xf = x0.clone()
      xf.requires_grad = compute_grad
      
      wf = f(xf)
 
      wf.backward(w0)

      # compare the solutions
      #######################################

      self.assertTrue(torch.norm(wm)>0.0)
      self.assertTrue(torch.norm(wf)>0.0)

      print('\n')
      print('fwd error = %.6e' % (torch.norm(wm-wf)/torch.norm(wf)))
      print('grad error = %.6e' % (torch.norm(xm.grad-xf.grad)/torch.norm(xf.grad)))
      print('\n')

      self.assertTrue(torch.norm(wm-wf)/torch.norm(wf)<=test_tol)
      self.assertTrue((torch.norm(xm.grad-xf.grad)/torch.norm(xf.grad))<=test_tol)

      print(len(list(f.parameters())),len(m_param_grad))
      for pf,pm_grad in zip(list(f.parameters()),m_param_grad):
        self.assertTrue(not pm_grad is None)

        print('p grad error = %.6e (norm=%.6e, shape=%s)' % (torch.norm(pf.grad-pm_grad)/torch.norm(pf.grad), torch.norm(pf.grad), pf.grad.shape))
        self.assertTrue(torch.norm(pf.grad-pm_grad)<=test_tol)
        
  # forwardPropSerial

if __name__ == '__main__':
  unittest.main()
