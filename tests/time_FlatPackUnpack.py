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

def time_packUnpack(sizes,ctm,device):
  in_tens = [torch.randn(s,device=device) for s in sizes]

  with ctm.timer('buffer_size'):
    buf_sz = utils.buffer_size(in_tens)

  with ctm.timer('pack_buffer'):
    buf = utils.pack_buffer(in_tens)

  assert(len(buf.shape)==1)
  assert(buf.shape[0]==buf_sz)

  out_tens = [torch.zeros(s,device=device) for s in sizes]

  with ctm.timer('unpack_buffer'):
    utils.unpack_buffer(out_tens,buf)

  for i,o in zip(in_tens,out_tens):
    assert(torch.norm(i-o).item()==0.0)
# end time_packUnpack

ctm = utils.ContextTimerManager()
device = torch.device('cpu')

mult = 10
sizes = [(9*mult,2*mult,3*mult),
         (2*mult,5*mult),
         (4*mult,1*mult,7*mult,9*mult)]

iters = 200
for i in range(0,iters):
  time_packUnpack(sizes,ctm,device)

print(ctm.getResultString())
