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

class DummyApp:
  def __init__(self,dtype,layer_data_size):
    self.dtype = dtype
    self.layer_data_size = layer_data_size

  def buildInit(self,t):
    # recoggnize that the default for pytorch is a 32 bit float...
    return torchbraid.BraidVector(torch.ones(4,5,dtype=self.dtype),0)

  def getTensorShapes(self):
    return [torch.Size(s) for s in [(4,5),(3,2,2,4),(1,3),(9,7,4)]]

  def getBufSize(self):
     return sizeof(int)+ (2+4+2+3)*sizeof(int)

  def getLayerDataSize(self):
     return self.layer_data_size
# end DummyApp

class TestLayerData:
  def __init__(self):
    self.s = { 'a':'some sort of data' + 'a'*10, 4: 2343}
    self.linear = torch.nn.Linear(10,10)

class TestTorchBraid(unittest.TestCase):
  def test_clone_init(self):
    app = DummyApp(float,0)

    clone_vec = cbs.cloneInitVector(app)
    clone = clone_vec.tensor()
    clone_sz = clone.size()

    self.assertEqual(len(clone_sz),2)
    self.assertEqual(clone_sz[0],4)
    self.assertEqual(clone_sz[1],5)
    self.assertEqual(torch.norm(clone).item(),np.sqrt(4.0*5.0))

  def test_clone_vector(self):
    app = DummyApp(float,0)

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
    layer_data_size = 27

    app = DummyApp(float,layer_data_size)
    shapes = app.getTensorShapes()

    self.assertEqual(layer_data_size,app.getLayerDataSize())

    a = torch.ones(shapes[0])
    b = torch.ones(shapes[1])
    c = torch.ones(shapes[2])
    d = torch.ones(shapes[3])

    bv = torchbraid.BraidVector((a,b),0)
    bv.addWeightTensors((c,d))

    num_tensors = len(shapes)

    data_shapes = 0
    data_size = 0
    for s in shapes:
      data_shapes += len(s)*sizeof_int
      data_size += s.numel()*sizeof_float

    sz = cbs.bufSize(app)

    total_size = ( sizeof_int                # level
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

    layer_data = TestLayerData()
    pickle_sz = tbutils.pickle_size(layer_data)

    app = DummyApp(torch.float,pickle_sz)
    shapes = app.getTensorShapes()

    a = torch.ones(shapes[0])
    b = torch.ones(shapes[1])
    c = torch.ones(shapes[2])
    d = torch.ones(shapes[3])

    bv_in = torchbraid.BraidVector((a,b),0)
    bv_in.addWeightTensors((c,d))
    bv_in.addLayerData(layer_data)

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
    self.assertEqual(bv_in.level(),bv_out.level())
    self.assertEqual(len(bv_in.allTensors()),len(bv_out.allTensors()))
    for i,o in zip(bv_in.allTensors(),bv_out.allTensors()):
      self.assertTrue(torch.norm(i-2.0*o).item()<5.0e-16)

    out_layer_data = bv_out.getLayerData()
    in_layer_data = bv_out.getLayerData()

    self.assertEqual(out_layer_data.s,in_layer_data.s)
    self.assertEqual(out_layer_data.linear,in_layer_data.linear)
    
if __name__ == '__main__':
  unittest.main()
