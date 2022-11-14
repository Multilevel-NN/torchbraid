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

from torchbraid.utils import getDevice
from mpi4py import MPI

def root_print(rank,s):
  if rank==0:
    print(s)

class ConvBlock(nn.Module):
  def __init__(self,dim,num_ch):
    super(ConvBlock, self).__init__()
    self.lin = nn.Conv1d(num_ch,num_ch,kernel_size=3,padding=1,bias=False)

  def forward(self, x):
    return F.relu(self.lin(x))
# end layer

def useCuda(comm):
  if torch.cuda.is_available() and torch.cuda.device_count()>=comm.Get_size():
    return True
  return False

class TestLayerParallel_MultiNODE(unittest.TestCase):

  def test_NoMGRIT(self):
    tolerance = 5e-7
    Tf = 4.0
    max_levels = 1
    max_iters = 1

    self.forwardBackwardProp(tolerance,Tf,max_levels,max_iters)

  def test_MGRIT(self):
    if useCuda(MPI.COMM_WORLD):
      return 

    tolerance = 5e-7
    Tf = 1.0
    max_levels = 3
    max_iters = 6 

    self.forwardBackwardProp(tolerance,Tf,max_levels,max_iters)

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

  def forwardBackwardProp(self,tolerance,Tf,max_levels,max_iters,check_grad=True,print_level=0):
    rank = MPI.COMM_WORLD.Get_rank()

    my_device,my_host = getDevice(MPI.COMM_WORLD) 

    root_print(rank,'\n')
    root_print(rank,self.id())

    dim = 10
    num_ch = 3
    num_samp = 8

    conv_block = lambda: ConvBlock(dim,num_ch)
    pool_block = lambda: nn.MaxPool1d(3)

    basic_block = [pool_block, conv_block,pool_block,conv_block, conv_block]
    num_steps   = [          1,         15,         1,         15,       16]

    # this is the torchbraid class being tested 
    #######################################
    parallel_net = torchbraid.LayerParallel(MPI.COMM_WORLD,basic_block,num_steps,Tf,max_fwd_levels=max_levels,max_bwd_levels=max_levels,max_iters=max_iters)
    parallel_net = parallel_net.to(my_device)
    parallel_net.setPrintLevel(print_level)
    parallel_net.setSkipDowncycle(False)

    # this is the reference torch "solution"
    #######################################
    serial_net = parallel_net.buildSequentialOnRoot()
    if serial_net is not None: # handle the cuda case
      serial_net = serial_net.to(my_device)

    # run forward/backward propgation
    xs = torch.rand(num_samp,num_ch,dim,device=my_device) # forward initial cond
    xs.requires_grad = check_grad

    ws = None
    if rank==0:
      ys = serial_net(xs)

      if check_grad:
        ws = torch.rand(ys.size(),device=my_device) # bacwards solution

        ys.backward(ws)
        adj_serial = xs.grad


    # propagation with torchbraid
    #######################################
    xp = xs.detach().clone()
    xp.requires_grad = check_grad

    yp = parallel_net(xp)
    yp_root = parallel_net.getFinalOnRoot(yp)

    if check_grad:
      wp = parallel_net.copyVectorFromRoot(ws)
      wp = wp.to(device=my_device)

      yp.backward(wp)
      adj_para_root = xp.grad
      param_grad_para = self.copyParameterGradToRoot(parallel_net,my_device)

    if rank==0:
      # check error
      forward_error = (torch.norm(ys-yp_root)/torch.norm(ys)).item()
      root_print(rank,f'Forward Error: {forward_error:e}')
      self.assertLessEqual(forward_error,tolerance,
                           "Relative error in the forward propgation, serial to parallel comparison.")

      if check_grad:
        # check adjoint
        backward_error = (torch.norm(adj_serial-adj_para_root)/torch.norm(adj_serial)).item()
        root_print(rank,f'Backward Error: {backward_error:e}')
        self.assertLessEqual(backward_error,tolerance,
                             "Relative error in the backward propgation, serial to parallel comparison.")

        param_errors = []
        param_norms = []
        param_grad_serial = [p.grad for p in serial_net.parameters()]
        for serial_pgrad,para_pgrad in zip(param_grad_serial,param_grad_para):
          self.assertTrue(not para_pgrad is None)
   
          # accumulate parameter errors for testing purposes
          pgrad_error = (torch.norm(serial_pgrad-para_pgrad)).item()
          param_errors += [pgrad_error]
          param_norms += [torch.norm(serial_pgrad).item()]

          #print(param_errors[-1],torch.norm(pf.grad),pf.grad.shape,torch.norm(pm_grad))
   
          # check the error conditions
          self.assertLessEqual(pgrad_error,tolerance,
                             "Relative error in the parameter gradient, serial to parallel comparison.")
   
        if len(param_errors)>0:
          print('p grad error (mean,stddev) = %.6e, %.6e' % (stats.mean(param_errors),stats.stdev(param_errors)))
          self.assertTrue(max(param_norms)>0.0)
# forwardBackwardProp 

if __name__ == '__main__':
  unittest.main()
