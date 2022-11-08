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

from mpi4py import MPI

def getDevice(comm):
  my_host    = torch.device('cpu')
  use_cuda = False
  if torch.cuda.is_available() and torch.cuda.device_count()>=comm.Get_size():
    if comm.Get_rank()==0:
      print('Using GPU Device')
    my_device  = torch.device(f'cuda:{comm.Get_rank()}')
    use_cuda = True
  elif torch.cuda.is_available() and torch.cuda.device_count()<comm.Get_size():
    if comm.Get_rank()==0:
      print('GPUs are not used, because MPI ranks are more than the device count, using CPU')
    my_device = my_host
  else:
    if comm.Get_rank()==0:
      print('No GPUs to be used, CPU only')
    my_device = my_host
  return my_device,my_host,use_cuda
# end getDevice

use_cuda = False
device = torch.device('cpu')

class DummyApp:
  def __init__(self,dtype,layer_data_size,use_cuda):
    self.dtype = dtype
    self.layer_data_size = 0
    self.timer_manager = tbutils.ContextTimerManager()
    self.use_cuda = use_cuda
    self.device = device
    self.gpu_direct_commu = False

  def buildInit(self,t):
    # recoggnize that the default for pytorch is a 32 bit float...
    return torchbraid.BraidVector(torch.ones(4,5,dtype=self.dtype))

  def getTensorShapes(self):
    return [torch.Size(s) for s in [(4,5),(3,2,2,4),(1,3),(9,7,4)]]

  def getBufSize(self):
     return sizeof(int)+ (2+4+2+3)*sizeof(int)

  def getLayerDataSize(self):
     return 0

  def timer(self,name):
    return self.timer_manager.timer("Dummy::"+name)

# end DummyApp

class TestLayerData:
  def __init__(self):
    self.s = { 'a':'some sort of data' + 'a'*10, 4: 2343}
    self.linear = torch.nn.Linear(10,10).to(device)

class TestTorchBraid(unittest.TestCase):
  def test_clone_init(self):
    app = DummyApp(float,0,use_cuda)
    clone_vec = cbs.cloneInitVector(app)
    clone = clone_vec.tensor()
    clone_sz = clone.size()

    self.assertEqual(len(clone_sz),2)
    self.assertEqual(clone_sz[0],4)
    self.assertEqual(clone_sz[1],5)
    self.assertEqual(torch.norm(clone).item(),np.sqrt(4.0*5.0))

  def test_clone_vector(self):
    app = DummyApp(float,0,use_cuda)

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
    layer_data_size = 0

    app = DummyApp(float,layer_data_size,use_cuda)
    shapes = app.getTensorShapes()

    self.assertEqual(layer_data_size,app.getLayerDataSize())

    a = torch.ones(shapes[0],device=device)
    b = torch.ones(shapes[1],device=device)
    c = torch.ones(shapes[2],device=device)
    d = torch.ones(shapes[3],device=device)

    bv = torchbraid.BraidVector((a,b))
    bv.addWeightTensors((c,d))

    num_tensors = len(shapes)

    data_shapes = 0
    data_size = 0
    for s in shapes:
      data_shapes += len(s)*sizeof_int
      data_size += s.numel()*sizeof_float

    sz = cbs.bufSize(app)

    total_size = ( sizeof_int                # floats
                 + sizeof_int                # num tensors
                 + sizeof_int                # num_weighttensors
                 + num_tensors*sizeof_int    # number of tensors dimensions and shapes
                 + data_shapes               # the shapes of each tensor
                 + data_size                 # the shapes of each tensor
                 + sizeof_int                # how much layer data (bytes)
                 + layer_data_size           # checkout of layer data
                 )

    self.assertEqual(sz,total_size)
  # end test_buff_size

  def test_buff_pack_unpack(self):
    sizeof_float = cbs.sizeof_float()
    sizeof_int   = cbs.sizeof_int()

    pickle_sz = 0

    app = DummyApp(torch.float,pickle_sz,use_cuda)
    shapes = app.getTensorShapes()

    a = torch.ones(shapes[0],device=device)
    b = torch.ones(shapes[1],device=device)
    c = torch.ones(shapes[2],device=device)
    d = torch.ones(shapes[3],device=device)

    bv_in = torchbraid.BraidVector((a,b))
    bv_in.addWeightTensors((c,d))

    # allocate space
    block = cbs.MemoryBlock(cbs.bufSize(app))

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
      self.assertTrue(torch.norm(i-2.0*o).item()<5.0e-16)

if __name__ == '__main__':
  device,_,use_cuda = getDevice(MPI.COMM_WORLD)
  unittest.main()
