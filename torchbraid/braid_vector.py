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

  def __init__(self,tensor,send_flag=False):
    BraidVector.instance += 1

    self.instance = BraidVector.instance
    self.weight_tensor_data_ = []
    self.send_flag_ = send_flag;

    self.stream = None

    if isinstance(tensor,torch.Tensor):
      self.tensor_data_ = (tensor,)
    elif isinstance(tensor,Iterable):
      # if the input is a tuple, that is the full data
      self.tensor_data_ = tuple(tensor)
    elif tensor==None:
      self.tensor_data_ = (tensor,)
    else:
      assert(False)

  def __del__(self):
    self.tensor_data_ = None
    self.weight_tensor_data_ = None

  def setStream(self,s):
    self.stream = s

  def hasStream(self):
    return self.stream is not None

  def syncStream(self):
    if self.hasStream():
      self.stream.synchronize()
      self.stream = None

  def addWeightTensors(self,weights):
    """
     
    """
    self.weight_tensor_data_ = list(weights)

  def releaseWeightTensors(self):
    self.weight_tensor_data_ = []

  def replaceTensor(self,tensor,i=0):
    """
    Replace the tensor. This is a shallow
    copy of the tensor. This method returns the old
    tensor object.
    """
    if isinstance(tensor,torch.Tensor):
      old_t = self.tensor_data_[i]
      tensor_lst = list(self.tensor_data_)
      tensor_lst[i] = tensor
      self.tensor_data_= tuple(tensor_lst)
    elif isinstance(tensor,Iterable):
      # pre-condition...Only tensors
      for t in tensor:
        assert(isinstance(t,torch.Tensor))

      old_t = self.tensor_data_

      # if the input is a tuple, that is the full data
      self.tensor_data_ = tuple(tensor)
    else:
      assert(False)

    return old_t

  def tensor(self,i=0):
    """
    Return a tensor from the tuple storage.
    Defaults to the first one (index 0)
    """
    return self.tensor_data_[i]

  def tensors(self):
    return self.tensor_data_

  def weightTensors(self):
    return self.weight_tensor_data_

  def allTensors(self):
    return list(self.tensor_data_) + self.weight_tensor_data_

  def getSendFlag(self):
    return self.send_flag_

  def setSendFlag(self,send_flag):
    self.send_flag_ = send_flag
  
  def clone(self):
    with torch.no_grad():
      tensors = [t.detach().clone() for t in self.tensors()]
      cl = BraidVector(tuple(tensors))

      # copy any weight tensors
      tensors = [t.detach() for t in self.weightTensors()]
      cl.addWeightTensors(tensors)

      cl.setSendFlag(self.getSendFlag())

    return cl
