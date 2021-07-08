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
import numpy as np

def buffer_size(tens):
  """
  Compute the size of the buffer need for a simple
  packing of this tensor, or a list of tensors.

  tens: Input tensor (single), or a list of tensors
  """

  if isinstance(tens,torch.Tensor):
    tens = [tens]    

  return sum([t.shape.numel() for t in tens if t is not None])
# end buffer_size 

def pack_buffer(tens):
  """
  Pack the list of tensors into a 1D array. This
  packing assumes that the unpack direction already
  has has sized tensors

  tens: Input tensor (single), or a list of tensors
  """

  if isinstance(tens,torch.Tensor):
    tens = [tens]    

  # flatten array and shove it into a flat buffer
  beg = 0
  end = 0
  buf = np.zeros(buffer_size(tens))
  for t in tens:
    if t is None:
      continue
    end += t.shape.numel()
    buf[beg:end] = t.view(-1)[:]
    beg = end

  return buf
# end pack_buffer

def unpack_buffer(tens,buf):
  """
  Unpack a 1D array into a list of tensors. 

  buf: 1D buffer to unpack from
  tens: Input tensor (single), or a list of tensors
  """

  if isinstance(tens,torch.Tensor):
    tens = [tens]    

  # flatten array and shove it into a flat buffer
  beg = 0
  end = 0
  for t in tens:
    if t is None:
      continue
    end += t.shape.numel()
    t.view(-1).numpy()[:] = buf[beg:end]
    beg = end

# end unpack_buffer
