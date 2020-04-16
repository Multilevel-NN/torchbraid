import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest
import sys
import numpy as np
import statistics as stats

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
    self.backForwardProp(dim,basic_block,x0,w0,max_levels,max_iters,test_tol=1e-16,prefix='linearNet_Exact')

    MPI.COMM_WORLD.barrier()
  # end test_linearNet_Exact

  def test_linearNet_Approx(self):
    dim = 2
    basic_block = lambda: LinearBlock(dim)

    x0 = torch.randn(5,dim) # forward initial cond
    w0 = torch.randn(5,dim) # adjoint initial cond
    max_levels = 3
    max_iters = 8
    self.backForwardProp(dim,basic_block,x0,w0,max_levels,max_iters,test_tol=1e-6,prefix='linearNet_Approx')

    MPI.COMM_WORLD.barrier()
  # end test_linearNet_Approx

  def test_reLUNet_Exact(self):
    dim = 2
    basic_block = lambda: ReLUBlock(dim)

    x0 = torch.randn(5,dim) # forward initial cond
    w0 = torch.randn(5,dim) # adjoint initial cond
    x0 = 12.0*torch.ones(5,dim) # forward initial cond
    w0 = 8.0*torch.ones(5,dim) # adjoint initial cond
    max_levels = 1
    max_iters = 1

    # this catch block, augments the 
    rank = MPI.COMM_WORLD.Get_rank()
    try:
      self.backForwardProp(dim,basic_block,x0,w0,max_levels,max_iters,test_tol=1e-16,prefix='reLUNet_Exact')
    except RuntimeError as err:
      raise RuntimeError("proc=%d) reLUNet_Exact..failure" % rank) from err

    MPI.COMM_WORLD.barrier()
  # end test_reLUNet_Exact

  def test_reLUNet_Approx(self):
    dim = 2
    basic_block = lambda: ReLUBlock(dim)

    x0 = torch.randn(5,dim) # forward initial cond
    w0 = torch.randn(5,dim) # adjoint initial cond
    x0 = 12.0*torch.ones(5,dim) # forward initial cond
    w0 = 8.0*torch.ones(5,dim) # adjoint initial cond
    max_levels = 3
    max_iters = 8
    # this catch block, augments the 
    rank = MPI.COMM_WORLD.Get_rank()
    try:
      self.backForwardProp(dim,basic_block,x0,w0,max_levels,max_iters,test_tol=1e-6,prefix='reLUNet_Approx')
    except RuntimeError as err:
      raise RuntimeError("proc=%d) reLUNet_Approx..failure" % rank) from err

    MPI.COMM_WORLD.barrier()
  # end test_reLUNet_Approx

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

  def backForwardProp(self,dim, basic_block,x0,w0,max_levels,max_iters,test_tol,prefix):
    Tf = 2.0
    num_steps = 4

    # this is the torchbraid class being tested 
    #######################################
    m = torchbraid.LayerParallel(MPI.COMM_WORLD,basic_block,num_steps,Tf,max_levels=max_levels,max_iters=max_iters)
    m.setPrintLevel(0)

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
      print('%s: fwd error = %.6e' % (prefix,torch.norm(wm-wf)/torch.norm(wf)))
      print('%s: grad error = %.6e' % (prefix,torch.norm(xm.grad-xf.grad)/torch.norm(xf.grad)))

      self.assertTrue(torch.norm(wm-wf)/torch.norm(wf)<=test_tol)
      self.assertTrue((torch.norm(xm.grad-xf.grad)/torch.norm(xf.grad))<=test_tol)

      param_errors = []
      for pf,pm_grad in zip(list(f.parameters()),m_param_grad):
        self.assertTrue(not pm_grad is None)

        # accumulate parameter errors for testing purposes
        param_errors += [(torch.norm(pf.grad-pm_grad)/torch.norm(pf.grad)).item()]
 
        # check the error conditions
        self.assertTrue(torch.norm(pf.grad-pm_grad)<=test_tol)

      if len(param_errors)>0:
        print('%s: p grad error (mean,stddev) = %.6e, %.6e' % (prefix,stats.mean(param_errors),stats.stdev(param_errors)))

      print('\n')
        
  # forwardPropSerial

if __name__ == '__main__':
  unittest.main()
