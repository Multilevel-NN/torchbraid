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

#from braid import *

include "./braid.pyx"
include "./torchbraid_callbacks.pyx"

#  a python level module
##########################################################

class BraidApp:

  def __init__(self,comm,local_num_steps,hidden_size,num_layers,Tf,max_levels,max_iters):
    # optional parameters
    self.max_levels  = max_levels
    self.max_iters   = max_iters
    self.print_level = 2
    self.nrelax      = 0
    self.cfactor     = 2
    self.skip_downcycle = 0

    local_num_steps = 1 # To make sure, set local_num_steps = 1
    self.mpi_comm        = comm
    self.Tf              = Tf
    self.local_num_steps = local_num_steps
    # self.num_steps       = local_num_steps
    self.num_steps       = local_num_steps*self.mpi_comm.Get_size()

    self.dt       = Tf/self.num_steps
    self.t0_local = self.mpi_comm.Get_rank()*local_num_steps*self.dt
    self.tf_local = (self.mpi_comm.Get_rank()+1.0)*local_num_steps*self.dt

    self.hidden_size = hidden_size
    self.num_layers = num_layers

    self.x_final = None
    self.shape0 = None
  
    comm          = self.getMPIComm()
    rnn_my_rank       = self.getMPIComm().Get_rank()
    num_ranks     = self.getMPIComm().Get_size()

    self.py_core = None

    # build up the core
    self.py_core = self.initCore()
  # end __init__

  def initCore(self):
    cdef braid_Core core
    cdef PyBraid_Core py_fwd_core
    cdef braid_Core fwd_core
    cdef double tstart
    cdef double tstop
    cdef int ntime
    cdef MPI.Comm comm = self.mpi_comm
    cdef int rank = self.mpi_comm.Get_rank()
    cdef braid_App app = <braid_App> self
    cdef braid_PtFcnStep  b_step  = <braid_PtFcnStep> my_step
    cdef braid_PtFcnInit  b_init  = <braid_PtFcnInit> my_init
    cdef braid_PtFcnClone b_clone = <braid_PtFcnClone> my_clone
    cdef braid_PtFcnFree  b_free  = <braid_PtFcnFree> my_free
    cdef braid_PtFcnSum   b_sum   = <braid_PtFcnSum> my_sum
    cdef braid_PtFcnSpatialNorm b_norm = <braid_PtFcnSpatialNorm> my_norm
    cdef braid_PtFcnAccess b_access = <braid_PtFcnAccess> my_access
    cdef braid_PtFcnBufSize b_bufsize = <braid_PtFcnBufSize> my_bufsize
    cdef braid_PtFcnBufPack b_bufpack = <braid_PtFcnBufPack> my_bufpack
    cdef braid_PtFcnBufUnpack b_bufunpack = <braid_PtFcnBufUnpack> my_bufunpack

    ntime = self.num_steps
    tstart = 0.0
    tstop = self.Tf

    braid_Init(comm.ob_mpi, comm.ob_mpi, 
               tstart, tstop, ntime, 
               app,
               b_step, b_init, 
               b_clone, b_free, 
               b_sum, b_norm, b_access, 
               b_bufsize, b_bufpack, b_bufunpack, 
               &core)

    # Set Braid options
    braid_SetMaxLevels(core, self.max_levels)
    braid_SetMaxIter(core, self.max_iters)
    braid_SetPrintLevel(core,self.print_level)
    braid_SetNRelax(core,-1,self.nrelax)
    braid_SetNRelax(core,0,0) # set F relax on fine grid
    braid_SetCFactor(core,-1,self.cfactor) # -1 implies chage on all levels
    braid_SetSkip(core,self.skip_downcycle)

    # store the c pointer
    py_core = PyBraid_Core()
    py_core.setCore(core)

    return py_core
  # end initCore

  def __del__(self):
    if self.py_core!=None:
      py_core = <PyBraid_Core> self.py_core
      core = py_core.getCore()

      # Destroy Braid Core C-Struct
      # FIXME: braid_Destroy(core) # this should be on
    # end core

  def getLayerDataSize(self):
    return 0

  def getTensorShapes(self):
    return self.shape0

  def setShape(self,shape):
    # the shape to use if non-exists for taking advantage of allocations in braid
    if isinstance(shape,torch.Size):
      self.shape0 = (shape,)
    else:
      self.shape0 = shape

  def runBraid(self,x):

    cdef PyBraid_Core py_core = <PyBraid_Core> self.py_core
    cdef braid_Core core = py_core.getCore()

    total_ranks   = self.mpi_comm.Get_size()
    comm_ = self.mpi_comm

    self.x = x
    
    h = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
    c = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

    # h = torch.ones(self.num_layers, x.size(0), self.hidden_size) * 77
    # c = torch.ones(self.num_layers, x.size(0), self.hidden_size)

    self.setInitial_g((h,c))
    # self.setInitial(x)

    # Run Braid (calls rnn_my_step -> eval(running basic_blocks) in RNN_torchbraid.py)
    braid_Drive(core)

    h_c  = self.getFinal()
    h_c = comm_.bcast(h_c,root=total_ranks-1)

    # print("Rank %d BraidApp -> runBraid() - end" % prefix_rank)

    return h_c

  def getCore(self):
    return self.py_core    
 
  def setPrintLevel(self,print_level):
    self.print_level = print_level

    core = (<PyBraid_Core> self.py_core).getCore()
    braid_SetPrintLevel(core,self.print_level)

  def setNumRelax(self,relax,level=-1):
    self.nrelax = relax

    core = (<PyBraid_Core> self.py_core).getCore()
    braid_SetNRelax(core,level,self.nrelax)

  def setCFactor(self,cfactor):
    self.cfactor = cfactor 

    core = (<PyBraid_Core> self.py_core).getCore()
    braid_SetCFactor(core,-1,self.cfactor) # -1 implies chage on all levels

  def setSkipDowncycle(self,skip):
    self.skip_downcycle = skip

    core = (<PyBraid_Core> self.py_core).getCore()
    braid_SetSkip(core,self.skip_downcycle)

  def setStorage(self,storage):
    core = (<PyBraid_Core> self.py_core).getCore()
    braid_SetStorage(core,storage)

  def setRevertedRanks(self,reverted):
    core = (<PyBraid_Core> self.py_core).getCore()
    braid_SetRevertedRanks(core,reverted)

  def getUVector(self,level,index):
    cdef braid_Core core = (<PyBraid_Core> self.py_core).getCore()
    cdef braid_BaseVector bv
    _braid_UGetVectorRef(core, level, index,&bv)

    return <object> bv.userVector

  def getMPIComm(self):
    return self.mpi_comm

  def getLocalTimeStepIndex(self,t,tf,level):
    return round((t-self.t0_local) / self.dt)

  def getGlobalTimeStepIndex(self,t,tf,level):
    return round(t / self.dt)

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
