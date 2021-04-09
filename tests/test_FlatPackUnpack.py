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

import unittest
import faulthandler
faulthandler.enable()

import time
import torch
import torchbraid.utils as utils

import numpy as np

class TestFlatPackUnpack(unittest.TestCase):

  def test_bufferSize(self):
     t0 = torch.zeros(9,2,3)
     t1 = torch.zeros(2,5)
     t2 = torch.zeros(4,1,7,9)

     self.assertEqual(utils.buffer_size(t0),9*2*3)
     self.assertEqual(utils.buffer_size(t1),2*5)
     self.assertEqual(utils.buffer_size(t2),4*1*7*9)

     self.assertEqual(utils.buffer_size([t0,t1,t2]),9*2*3+2*5+4*1*7*9)

  def test_packUnpack(self):
     t0 = torch.randn(9,2,3)
     t1 = torch.randn(2,5)
     t2 = torch.randn(4,1,7,9)

     buf_sz = utils.buffer_size([t0,t1,t2])
     buf = utils.pack_buffer([t0,t1,t2])

     self.assertEqual(len(buf.shape),1)
     self.assertEqual(buf.shape[0],buf_sz)

     t0_u = torch.zeros(9,2,3)
     t1_u = torch.zeros(2,5)
     t2_u = torch.zeros(4,1,7,9)

     utils.unpack_buffer([t0_u,t1_u,t2_u],buf)

     self.assertEqual(torch.norm(t0_u-t0).item(),0.0)
     self.assertEqual(torch.norm(t1_u-t1).item(),0.0)
     self.assertEqual(torch.norm(t2_u-t2).item(),0.0)

if __name__ == '__main__':
  unittest.main()
