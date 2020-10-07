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
from collections.abc import Iterable

from mpi4py import MPI

class BraidVector:
  instance = -1 

  def __init__(self,tensor,level):
    BraidVector.instance += 1

    self.instance = BraidVector.instance

    s = ''
    if isinstance(tensor,Iterable):
      s += 'iterable'

    if isinstance(tensor,torch.Tensor):
      self.tensor_data_ = (tensor,)
    elif isinstance(tensor,Iterable):
      # pre-condition...Only tensors
      for t in tensor:
        assert(isinstance(t,torch.Tensor))

      # if the input is a tuple, that is the full data
      self.tensor_data_ = tuple(tensor)
    elif tensor==None:
      self.tensor_data_ = (tensor,)
    else:
      assert(False)
        
    self.level_  = level

  def __del__(self):
    self.tensor_data_ = None

  def replaceTensor(self,t,i=0):
    """
    Replace the teensor. This is a shallow
    copy of the tensor. This method returns the old
    tensor object.
    """
    assert(isinstance(t,torch.Tensor))
    old_t = self.tensor_data_[i]
    tensor_lst = list(self.tensor_data_)
    tensor_lst[i] = t
    self.tensor_data_= tuple(tensor_lst)

    return old_t

  def tensor(self,i=0):
    """
    Return a tensor from the tuple storage.
    Defaults to the first one (index 0)
    """
    return self.tensor_data_[i]

  def tensors(self):
    return self.tensor_data_

  def level(self):
    return self.level_
  
  def clone(self):
    tensors = [t.detach().clone() for t in self.tensors()]
    cl = BraidVector(tuple(tensors),self.level())
    return cl
