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
import unittest
import sys
import traceback
import numpy as np
import statistics as stats

import torchbraid

import faulthandler
faulthandler.enable()

from torchbraid.utils import getDevice
from mpi4py import MPI

def output_exception():
  s = traceback.format_exc()
  print('\n**** TEST GENERIC Exception ****\n{}'.format(s))

class ReLUBlock(nn.Module):
  def __init__(self,dim=10):
    super(ReLUBlock, self).__init__()
    self.lin = nn.Linear(dim, dim,bias=True)

    w = 2.3*torch.ones(dim,dim)
    self.lin.weight = torch.nn.Parameter(w)

    b = -1.22*torch.ones(dim)
    self.lin.bias = torch.nn.Parameter(b)

  def forward(self, x):
    return F.relu(self.lin(x))
# end layer

class TestGradUpdate(unittest.TestCase):

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

    self.backForwardProp(dim,basic_block,x0,w0,max_levels,max_iters,test_tol=1e-6,prefix='reLUNet_Approx')
  # end test_reLUNet_Approx

  def copyParameterGradToRoot(self,m,device):
    comm     = m.getMPIComm()
    my_rank  = m.getMPIComm().Get_rank()
    num_proc = m.getMPIComm().Get_size()
 
    params = [p.grad for p in list(m.parameters())]

    if len(params)==0:
      return params

    if my_rank==0:
      for i in range(1,num_proc):
        remote_p = comm.recv(source=i,tag=77)
        remote_p = [p.to(device) for p in remote_p]
        params.extend(remote_p)

      return params
    else:
      params_cpu = [p.cpu() for p in params]
      comm.send(params_cpu,dest=0,tag=77)
      return None
  # end copyParametersToRoot

  def backForwardProp(self,dim, basic_block,x0,w0,max_levels,max_iters,test_tol,prefix,ref_pair=None):
    Tf = 2.0
    num_procs = MPI.COMM_WORLD.Get_size()
    num_steps = 4*num_procs

    # figure out the whole GPU situation
    my_device,my_host = getDevice(MPI.COMM_WORLD) 

    x0 = x0.to(my_device)
    w0 = w0.to(my_device)

    # this is the torchbraid class being tested 
    #######################################
    m = torchbraid.LayerParallel(MPI.COMM_WORLD,basic_block,num_steps,Tf,max_fwd_levels=max_levels,max_bwd_levels=max_levels,max_iters=max_iters,spatial_ref_pair=ref_pair)
    m = m.to(my_device)
    m.setPrintLevel(0)

    w0 = m.copyVectorFromRoot(w0)

    # this is the reference torch "solution"
    #######################################
    dt = Tf/num_steps
    f = m.buildSequentialOnRoot()
    if f is not None: # handle the cuda case
      f = f.to(my_device)

    # run forward/backward propgation
    lr = 1e-3

    # propogation with torchbraid
    #######################################
    xm = x0.clone()
    xm.requires_grad = True

    wm = m(xm)
    wm.backward(w0)
    wm0 = m.getFinalOnRoot(wm)

    with torch.no_grad():
      for p in m.parameters(): p -= p.grad * lr
    m.zero_grad()

    if xm.grad is not None:
      xm.grad.zero_()
    wm = m(xm)
    wm.backward(w0)

    m_param_grad = self.copyParameterGradToRoot(m,my_device)
    wm = m.getFinalOnRoot(wm)

    # print time results
    timer_str = m.getTimersString() 

    # check some values
    if m.getMPIComm().Get_rank()==0:

      # this is too much to print out every test run, but I'd like to make sure the
      # code is execueted
      self.assertTrue(len(timer_str)>0)
 
      compute_grad = True

      # propogation with torch
      #######################################
      xf = x0.clone()
      xf.requires_grad = compute_grad
      
      wf = f(xf)
      wf.backward(w0)
 
      with torch.no_grad():
        for p in f.parameters(): 
          p -= p.grad * lr
        f.zero_grad()
     
      xf.grad.zero_()
      wf = f(xf)
      wf.backward(w0)

      # compare the solutions
      #######################################

      self.assertTrue(torch.norm(wm)>0.0)
      self.assertTrue(torch.norm(wf)>0.0)

      print('\n')
      print('%s: fwd error = %.6e (%.6e, %.6e)' % (prefix,torch.norm(wm-wf)/torch.norm(wf),torch.norm(wf),torch.norm(wm)))
      print('%s: grad error = %.6e (%.6e, %.6e)' % (prefix,torch.norm(xm.grad-xf.grad)/torch.norm(xf.grad),torch.norm(xf.grad),torch.norm(xm.grad)))

      self.assertTrue(torch.norm(wm-wf)/torch.norm(wf)<=test_tol)
      self.assertTrue((torch.norm(xm.grad-xf.grad)/torch.norm(xf.grad))<=test_tol)

      param_errors = []
      for pf,pm_grad in zip(list(f.parameters()),m_param_grad):
        self.assertTrue(not pm_grad is None)

        # accumulate parameter errors for testing purposes
        param_errors += [(torch.norm(pf.grad-pm_grad)/(1e-15+torch.norm(pf.grad))).item()]
 
        # check the error conditions
        #print('%s: p_grad error = %.6e (%.6e %.6e)' % (prefix,torch.norm(pf.grad-pm_grad),torch.norm(pf.grad),torch.norm(pm_grad)))
        #sys.stdout.flush()
        self.assertTrue(torch.norm(pf.grad-pm_grad)<=test_tol)

      if len(param_errors)>0:
        print('%s: p grad error (mean,stddev) = %.6e, %.6e' % (prefix,stats.mean(param_errors),stats.stdev(param_errors)))

      print('\n')
        
  # forwardPropSerial

if __name__ == '__main__':
  unittest.main()
