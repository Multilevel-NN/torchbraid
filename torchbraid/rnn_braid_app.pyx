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

from braid_vector import BraidVector

cimport mpi4py.MPI as MPI

import torchbraid_app as parent

include "./braid.pyx"

#  a python level module
##########################################################

class BraidApp(parent.BraidApp):

  def __init__(self,comm,local_num_steps,hidden_size,num_layers,Tf,max_levels,max_iters):
    parent.BraidApp.__init__(self,'RNN',comm,local_num_steps,Tf,max_levels,max_iters,spatial_ref_pair=None,require_storage=True)

    self.hidden_size = hidden_size
    self.num_layers = num_layers
  # end __init__

  def getTensorShapes(self):
    return self.shape0

  def runBraid(self,x,h_c=None):

    cdef PyBraid_Core py_core = <PyBraid_Core> self.py_core
    cdef braid_Core core = py_core.getCore()

    total_ranks   = self.mpi_comm.Get_size()
    comm_ = self.mpi_comm

    assert(x.shape[1]==self.local_num_steps)

    self.x = x
    
    if h_c is None:
      h = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
      c = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
    else:
      h = h_c[0]
      c = h_c[1]

    self.setInitial_g((h,c))

    # Run Braid (calls rnn_my_step -> eval(running basic_blocks) in RNN_torchbraid.py)
    braid_Drive(core)

    h_c  = self.getFinal()
    h_c = comm_.bcast(h_c,root=total_ranks-1)

    # print("Rank %d BraidApp -> runBraid() - end" % prefix_rank)

    return h_c

  def setInitial_g(self,g0):

    cdef braid_Core core = (<PyBraid_Core> self.py_core).getCore()
    cdef braid_BaseVector bv 

    self.g0 = BraidVector(g0,0)

    # set the appropriate initial condition
    if core.warm_restart:
      _braid_UGetVectorRef(core, 0, 0, &bv);
      if not (bv is NULL):
        py_bv = <object> bv.userVector
        py_bv.tensor_ = g0

    # print("Rank %d BraidApp -> setInitial_g() - end" % prefix_rank)

  def buildInit(self,t):

    g = self.g0.clone()
    if t>0:
      t_h,t_c = g.tensors()
      t_h[:] = 0.0
      t_c[:] = 0.0

    # print("Rank %d BraidApp -> buildInit() - end" % prefix_rank)
    return g

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

class BraidBackApp(parent.BraidApp):

  def __init__(self,prefix_str,comm,local_num_steps,Tf,max_levels,max_iters,
               spatial_ref_pair=None,require_storage=False):
    parent.BraidApp.__init__(self,prefix_str,comm,local_num_steps,Tf,max_levels,max_iters,spatial_ref_pair,require_storage)
  # end __init__

  def access(self,t,u):
    if t==self.Tf:
      # not sure why this requires a clone
      # if this needs only one processor
      # it could be a problem in the future
      if self.getMPIComm().Get_size()>1:
        self.x_final = u.tensors()
      else:
        self.x_final = u.clone().tensors()

# end BraidApp
