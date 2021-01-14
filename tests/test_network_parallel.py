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

class ConvBlock(nn.Module):
  def __init__(self,dim,num_ch):
    super(ConvBlock, self).__init__()
    self.lin = nn.Conv1d(num_ch,num_ch,kernel_size=3,padding=1,bias=False)

  def forward(self, x):
    return self.lin(x)
# end layer

class TestTorchBraid(unittest.TestCase):
  def test_distributeFromRoot(self):
    # dummy class to test distribution
    class Network:
      def children(self):
        return [i for i in range(13)]
   
    comm = MPI.COMM_WORLD
    network = Network()

    size = comm.Get_size()
    rank = comm.Get_rank()

    result = torchbraid.distributeNetworkFromRoot(comm,network)

    # print('  %d) ' % comm.Get_rank(),result)

    if size==1:
      self.assertEqual(result,list(range(0,13)))
    elif size==2:
      if rank==0: self.assertEqual(result,list(range(0,7)))
      if rank==1: self.assertEqual(result,list(range(7,13)))
    elif size==3:
      if rank==0: self.assertEqual(result,list(range(0,5)))
      if rank==1: self.assertEqual(result,list(range(5,9)))
      if rank==2: self.assertEqual(result,list(range(9,13)))
    elif size==4:
      if rank==0: self.assertEqual(result,list(range(0,4)))
      if rank==1: self.assertEqual(result,list(range(4,7)))
      if rank==2: self.assertEqual(result,list(range(7,10)))
      if rank==3: self.assertEqual(result,list(range(10,13)))
    else:
      assert(False) # can't run on more than four ranks
  # end test_distributeFromRoot

  def test_linearNet_Exact(self):
    dim = 2
    basic_block = lambda: LinearBlock(dim)

    x0 = torch.randn(5,dim) # forward initial cond
    w0 = torch.randn(5,dim) # adjoint initial cond
    max_levels = 1
    max_iters = 1
    self.backForwardProp(dim,basic_block,x0,w0,max_levels,max_iters,test_tol=1e-16,prefix='linearNet_Exact',check_grad=True)

    MPI.COMM_WORLD.barrier()
  # end test_linearNet_Exact

#   def test_linearNet_Approx(self):
#     dim = 2
#     basic_block = lambda: LinearBlock(dim)
# 
#     x0 = torch.randn(5,dim) # forward initial cond
#     w0 = torch.randn(5,dim) # adjoint initial cond
#     max_levels = 3
#     max_iters = 8
#     self.backForwardProp(dim,basic_block,x0,w0,max_levels,max_iters,test_tol=1e-6,prefix='linearNet_Approx')
# 
#     MPI.COMM_WORLD.barrier()
#   # end test_linearNet_Approx
# 
  def test_reLUNet_Exact(self):
    dim = 2
    basic_block = lambda: ReLUBlock(dim)

    x0 = 12.0*torch.ones(5,dim) # forward initial cond
    w0 = 3.0*torch.ones(5,dim) # adjoint initial cond
    max_levels = 1
    max_iters = 1

    # this catch block, augments the 
    rank = MPI.COMM_WORLD.Get_rank()
    self.backForwardProp(dim,basic_block,x0,w0,max_levels,max_iters,test_tol=1e-16,prefix='reLUNet_Exact',check_grad=True)

    MPI.COMM_WORLD.barrier()
  # end test_reLUNet_Exact
 
  def test_convNet_Exact(self):
    dim = 128
    num_ch = 1
    num_samp = 1
    basic_block = lambda: ConvBlock(dim,num_ch)

    u = torch.linspace(0.0,1.0,dim)
    x0 = torch.zeros(num_samp,num_ch,dim) # forward initial cond
    w0 = torch.zeros(num_samp,num_ch,dim) # adjoint initial cond
    for ch in range(num_ch):
      for samp in range(num_samp):
        x0[samp,ch,:] = torch.sin(2.0*np.pi*0.5*(ch+1.0)*(u-1.0/(samp+1.0)))
        w0[samp,ch,:] = torch.cos(2.0*np.pi*0.5*(ch+1.0)*(u-1.0/(samp+1.0))) 
    # build a relatively smooth initial guess

    max_levels = 1
    max_iters = 1

    # this catch block, augments the 
    rank = MPI.COMM_WORLD.Get_rank()
    self.backForwardProp(dim,basic_block,x0,w0,max_levels,max_iters,test_tol=1e-16,prefix='convNet_Exact',check_grad=True)

    MPI.COMM_WORLD.barrier()
  # end test_reLUNet_Exact
# 
#   def test_reLUNet_Approx(self):
#     dim = 2
#     basic_block = lambda: ReLUBlock(dim)
# 
#     x0 = 12.0*torch.ones(5,dim) # forward initial cond
#     w0 = 3.0*torch.ones(5,dim) # adjoint initial cond
#     max_levels = 3
#     max_iters = 8
#     # this catch block, augments the 
#     rank = MPI.COMM_WORLD.Get_rank()
#     try:
#       self.backForwardProp(dim,basic_block,x0,w0,max_levels,max_iters,test_tol=1e-6,prefix='reLUNet_Approx')
#     except RuntimeError as err:
#       raise RuntimeError("proc=%d) reLUNet_Approx..failure" % rank) from err
# 
#     MPI.COMM_WORLD.barrier()
#   # end test_reLUNet_Approx

  def copyParameterGradToRoot(self,m):
    comm     = m.getMPIComm()
    my_rank  = m.getMPIComm().Get_rank()
    num_proc = m.getMPIComm().Get_size()
 
    params = [p.grad for p in list(m.parameters())]

    return params

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

  def backForwardProp(self,dim, basic_block,x0,w0,max_levels,max_iters,test_tol,prefix,ref_pair=None,check_grad=True,num_steps=4,print_level=0):

    layers = [basic_block() for _ in range(num_steps)]

    # this is the reference torch "solution"
    #######################################
    f = torch.nn.Sequential(*layers)

    layers = torchbraid.distributeNetworkFromRoot(MPI.COMM_WORLD,f)
      
    # this is the torchbraid class being tested 
    #######################################
    m = torchbraid.NetworkParallel(MPI.COMM_WORLD,layers,max_levels=max_levels,max_iters=max_iters)

    m.setPrintLevel(print_level)
    m.setSkipDowncycle(False)

    w0 = m.copyVectorFromRoot(w0)

    # run forward/backward propgation

    # propogation with torchbraid
    #######################################
    xm = x0.clone()
    xm.requires_grad = check_grad

    wm = m(xm)

    if check_grad:
      wm.backward(w0)
      m_param_grad = self.copyParameterGradToRoot(m)

    wm = m.getFinalOnRoot(wm)

    # print time results
    timer_str = m.getTimersString() 

    # check some values
    if m.getMPIComm().Get_rank()==0:

      # this is too much to print out every test run, but I'd like to make sure the
      # code is execueted
      self.assertTrue(len(timer_str)>0)
 
      # propogation with torch
      #######################################
      xf = x0.clone()
      xf.requires_grad = check_grad
      
      wf = f(xf)
 
      # compare the solutions
      #######################################

      self.assertTrue(torch.norm(wm)>0.0)
      self.assertTrue(torch.norm(wf)>0.0)

      print('\n')
      print('%s: fwd error = %.6e' % (prefix,torch.norm(wm-wf)/torch.norm(wf)))

      self.assertTrue(torch.norm(wm-wf)/torch.norm(wf)<=test_tol)

      if check_grad:
        wf.backward(w0)
        print('%s: grad error = %.6e' % (prefix,torch.norm(xm.grad-xf.grad)/torch.norm(xf.grad)))
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

#import sys
#
#def trace(frame, event, arg):
#    print("%s, %s:%d" % (event, frame.f_code.co_filename, frame.f_lineno))
#    return trace
#
#sys.settrace(trace)

if __name__ == '__main__':
  unittest.main()
