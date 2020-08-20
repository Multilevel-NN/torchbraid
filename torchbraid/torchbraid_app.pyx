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

from cpython.ref cimport PyObject

from mpi4py import MPI
cimport mpi4py.MPI as MPI
cimport mpi4py.libmpi as libmpi

ctypedef PyObject _braid_App_struct 
ctypedef _braid_App_struct* braid_App

class BraidVector:
  def __init__(self,tensor,level):
    self.tensor_ = tensor 
    self.level_  = level
    self.time_   = np.nan # are we using this???

  def tensor(self):
    return self.tensor_

  def level(self):
    return self.level_
  
  def clone(self):
    cl = BraidVector(self.tensor().clone(),self.level())
    return cl

  def setTime(self,t):
    self.time_ = t

  def getTime(self):
    return self.time_

ctypedef PyObject _braid_Vector_struct
ctypedef _braid_Vector_struct *braid_Vector

## to supress a warning from numpy
#cdef extern from *:
#  """
#  #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#  """

include "./braid.pyx"
include "./torchbraid_callbacks.pyx"

#  a python level module
##########################################################

cdef class MPIData:
  cdef MPI.Comm comm
  cdef int rank
  cdef int size

  def __cinit__(self,comm):
    self.comm = comm
    self.rank = self.comm.Get_rank()
    self.size = self.comm.Get_size()

  def getComm(self):
    return self.comm

  def getRank(self):
    return self.rank 

  def getSize(self):
    return self.size
# helper class for the MPI communicator

##############################################################

class BraidApp:

  def __init__(self,comm,local_num_steps,Tf,max_levels,max_iters,
               spatial_ref_pair=None):
    # optional parameters
    self.max_levels  = max_levels
    self.max_iters   = max_iters
    self.print_level = 2
    self.nrelax      = 0
    self.cfactor     = 2
    self.skip_downcycle = 0

    self.mpi_data        = MPIData(comm)
    self.Tf              = Tf
    self.local_num_steps = local_num_steps
    self.num_steps       = local_num_steps*self.mpi_data.getSize()

    self.dt       = Tf/self.num_steps
    self.t0_local = self.mpi_data.getRank()*local_num_steps*self.dt
    self.tf_local = (self.mpi_data.getRank()+1.0)*local_num_steps*self.dt

    self.x_final = None
  
    comm          = self.getMPIData().getComm()
    my_rank       = self.getMPIData().getRank()
    num_ranks     = self.getMPIData().getSize()

    self.py_core = None

    self.spatial_mg = False
    self.spatial_ref_pair = spatial_ref_pair
    if spatial_ref_pair!=None:
      c,r = spatial_ref_pair
      self.spatial_coarse = c
      self.spatial_refine = r
      self.spatial_mg = True
    # turn on spatial multigrid

    # build up the core
    self.py_core = self.initCore()

    self.first = True
  # end __init__

  def initCore(self):
    cdef braid_Core core
    cdef PyBraid_Core py_fwd_core
    cdef braid_Core fwd_core
    cdef double tstart
    cdef double tstop
    cdef int ntime
    cdef MPI.Comm comm = self.mpi_data.getComm()
    cdef int rank = self.mpi_data.getRank()
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
    cdef braid_PtFcnSCoarsen b_coarsen = <braid_PtFcnSCoarsen> my_coarsen
    cdef braid_PtFcnSRefine b_refine = <braid_PtFcnSRefine> my_refine

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

    if self.spatial_mg:
      braid_SetSpatialCoarsen(core,b_coarsen)
      braid_SetSpatialRefine(core,b_refine)
    # end if refinement_on

    # Set Braid options
    braid_SetMaxLevels(core, self.max_levels)
    braid_SetMaxIter(core, self.max_iters)
    braid_SetPrintLevel(core,self.print_level)
    braid_SetNRelax(core,-1,self.nrelax)
    braid_SetNRelax(core,0,0) # set F relax on fine grid
    braid_SetCFactor(core,-1,self.cfactor) # -1 implies chage on all levels
    if self.skip_downcycle==0:
      braid_SetSkip(core,0)
    else:
      braid_SetSkip(core,1)

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

  def runBraid(self,x):
    cdef PyBraid_Core py_core = <PyBraid_Core> self.py_core
    cdef braid_Core core = py_core.getCore()

    self.setInitial(x)
 
    # Run Braid
    if not self.first:
      _braid_InitGuess(core,0)
      self.first = False
    braid_Drive(core) # my_step -> App:eval -> resnet "basic block"

    return self.getFinal()

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
    if skip:
      self.skip_downcycle = 1 
    else:
      self.skip_downcycle = 0 

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

  def getMPIData(self):
    return self.mpi_data

  def getLocalTimeStepIndex(self,t,tf,level):
    return round((t-self.t0_local) / self.dt)

  def getGlobalTimeStepIndex(self,t,tf,level):
    return round(t / self.dt)

  def setInitial(self,x0):
    cdef braid_Core core = (<PyBraid_Core> self.py_core).getCore()
    cdef braid_BaseVector bv 

    self.x0 = BraidVector(x0,0)

    # set the appropriate initial condition
    if core.warm_restart:
      _braid_UGetVectorRef(core, 0, 0, &bv);
      if not (bv is NULL):
        py_bv = <object> bv.userVector
        py_bv.tensor_ = x0


  def buildInit(self,t):
    x = self.x0.clone()
    if t>0:
      t_x = x.tensor()
      t_x[:] = 0.0
    return x

  def access(self,t,u):
    if t==self.Tf:
      self.x_final = u.clone()

  def getFinal(self):
    if self.x_final==None:
      return None

    # assert the level
    assert(self.x_final.level()==0)
    return self.x_final.tensor()

# end BraidApp
