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
import math
import unittest
import numpy as np

import torchbraid
import torchbraid.utils as tbutils

import test_cbs as cbs

from torchbraid.utils import getDevice
from mpi4py import MPI

use_cuda = False
device = torch.device('cpu')

class DummyApp:
  def __init__(self,dtype,use_cuda):
    self.dtype = dtype
    self.timer_manager = tbutils.ContextTimerManager()
    self.use_cuda = use_cuda
    self.device = device
    self.gpu_direct_commu = True
    self.buffer = []

  def buildInit(self,t):
    # recoggnize that the default for pytorch is a 32 bit float...
    return torchbraid.BraidVector(torch.ones(4,5,dtype=self.dtype))

  def getFeatureShapes(self,tidx,level):
    return [torch.Size(s) for s in [(4,5),(3,2,2,4)]]

  def getParameterShapes(self,tidx,level):
    return [torch.Size(s) for s in [(1,3),(9,7,4)]]

  def getBufSize(self):
     return sizeof(int)+ (2+4+2+3)*sizeof(int)

  def timer(self,name):
    return self.timer_manager.timer("Dummy::"+name)

  def addBufferEntry(self, tensor):
    self.buffer.append(tensor)
    return self.buffer[-1].data_ptr()

  def getBuffer(self, addr):
    for i in range(len(self.buffer)):
      dataPtr = self.buffer[i].data_ptr()
      if dataPtr == addr:
        return self.buffer[i]

    raise Exception('Buffer not found')

  def removeBufferEntry(self, addr):
    self.buffer = [item for item in self.buffer if item.data_ptr() != addr]

# end DummyApp

class TestTorchBraid(unittest.TestCase):
  def test_clone_init(self):
    app = DummyApp(float,use_cuda)
    clone_vec = cbs.cloneInitVector(app)
    clone = clone_vec.tensor()
    clone_sz = clone.size()

    self.assertEqual(len(clone_sz),2)
    self.assertEqual(clone_sz[0],4)
    self.assertEqual(clone_sz[1],5)
    self.assertEqual(torch.norm(clone).item(),np.sqrt(4.0*5.0))

  def test_clone_vector(self):
    app = DummyApp(float,use_cuda)

    vec = app.buildInit(0.0)
    ten = vec.tensor() 
    ten *= 2.0

    clone_vec = cbs.cloneVector(app,vec)
    clone = clone_vec.tensor()
    clone_sz = clone.size()

    self.assertEqual(len(clone_sz),2)
    self.assertEqual(clone_sz[0],4)
    self.assertEqual(clone_sz[1],5)
    self.assertEqual(torch.norm(clone).item(),np.sqrt(4.0*4.0*5.0))
  # end test_clone

  def test_buff_size(self):
    sizeof_float = cbs.sizeof_float()
    sizeof_int   = cbs.sizeof_int()

    app = DummyApp(float,use_cuda)
    shapes = app.getFeatureShapes(0,0) + app.getParameterShapes(0,0)

    a = torch.ones(shapes[0],device=device)
    b = torch.ones(shapes[1],device=device)
    c = torch.ones(shapes[2],device=device)
    d = torch.ones(shapes[3],device=device)

    bv = torchbraid.BraidVector((a,b))
    bv.addWeightTensors((c,d))

    num_tensors = len(shapes)

    data_size = 0
    for s in shapes:
      data_size += s.numel()
    data_size *= sizeof_float

    sz = cbs.bufSize(app)

    total_size = data_size

    self.assertEqual(sz,total_size)
  # end test_buff_size

  def test_buff_pack_unpack(self):
    sizeof_float = cbs.sizeof_float()
    sizeof_int   = cbs.sizeof_int()

    app = DummyApp(float,use_cuda)
    shapes = app.getFeatureShapes(0,0) + app.getParameterShapes(0,0)

    a = 1.*torch.ones(shapes[0],device=device)
    b = 2.*torch.ones(shapes[1],device=device)
    c = 3.*torch.ones(shapes[2],device=device)
    d = 4.*torch.ones(shapes[3],device=device)

    bv_in = torchbraid.BraidVector((a,b))
    bv_in.addWeightTensors((c,d))

    # allocate space
    block = cbs.MemoryBlock(app,cbs.bufSize(app))

    # communicate in/out
    cbs.pack(app,bv_in,block,0)

    # we want to make sure we are not just blindly copying
    # memory
    a += 1.0
    b += 1.0
    c += 1.0
    d += 1.0

    bv_out = cbs.unpack(app,block)

    # check the answers
    self.assertEqual(len(bv_in.allTensors()),len(bv_out.allTensors()))
    for i,o in zip(bv_in.allTensors(),bv_out.allTensors()):
      self.assertTrue(torch.norm(i-1.0-o).item()<5.0e-16)

if __name__ == '__main__':
  device,host_device = getDevice(MPI.COMM_WORLD)
  use_cuda = (device.type=='cuda')
  print(f'USE CUDA? = {use_cuda}')
  unittest.main()
