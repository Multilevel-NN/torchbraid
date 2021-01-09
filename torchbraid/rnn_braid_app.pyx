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

# cython: profile=True
# cython: linetrace=True

import torch
import numpy as np
import traceback

from braid_vector import BraidVector

cimport mpi4py.MPI as MPI

import torchbraid_app as parent

include "./braid.pyx"

#  a python level module
##########################################################

def output_exception(label):
  s = traceback.format_exc()
  print('\n**** Torchbraid Callbacks::{} Exception ****\n{}'.format(label,s))

class FwdRNNBraidApp(parent.BraidApp):

  def __init__(self,comm,local_num_steps,Tf,max_levels,max_iters):
    parent.BraidApp.__init__(self,'RNN',comm,local_num_steps,Tf,max_levels,max_iters,spatial_ref_pair=None,require_storage=True)

  # end __init__

  def getTensorShapes(self):
    return self.shape0

  def access(self,t,u):

    if t==self.Tf:
      self.x_final = u.clone()

    # print("Rank %d BraidApp -> access() - end" % prefix_rank)

  def getFinal(self):

    if self.x_final==None:
      return None
      
    # assert the level
    assert(self.x_final.level()==0)
    x_final_tensors = self.x_final.tensors()

    # print("Rank %d BraidApp -> getFinal() - end" % prefix_rank)
    return x_final_tensors

# end BraidApp

class BwdRNNBraidApp(parent.BraidApp):

  def __init__(self,prefix_str,comm,local_num_steps,Tf,max_levels,max_iters,
               spatial_ref_pair=None,require_storage=False):
    parent.BraidApp.__init__(self,prefix_str,comm,local_num_steps,Tf,max_levels,max_iters,spatial_ref_pair,require_storage)
  # end __init__

  def access(self,t,u):
    try:
      if t==self.Tf:
        # not sure why this requires a clone
        # if this needs only one processor
        # it could be a problem in the future
        if self.getMPIComm().Get_size()>1:
          self.x_final = u.tensors()
        else:
          self.x_final = u.clone().tensors()
    except:
      output_exception('BraidBackApp:access')

# end BraidApp
