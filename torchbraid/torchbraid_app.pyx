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

include "./braid.pyx"
include "./torchbraid_callbacks.pyx"

#  a python level module
##########################################################

def output_exception(label):
  s = traceback.format_exc()
  print('\n**** Torchbraid Callbacks::{} Exception ****\n{}'.format(label,s))

class BraidApp:

  def __init__(self,prefix_str,comm,local_num_steps,Tf,max_levels,max_iters,
               spatial_ref_pair=None,require_storage=False,abs_tol=1e-12):

    self.prefix_str = prefix_str # prefix string for helping to debug hopefully
    self.tb_print_level = 0      # set print level internally to zero

    # optional parameters
    self.max_levels  = max_levels
    self.max_iters   = max_iters
    self.print_level = 2
    self.nrelax      = 0
    self.cfactor     = 2
    self.skip_downcycle = 0
    self.require_storage = require_storage
    self.abs_tol = abs_tol

    self.mpi_comm        = comm
    self.Tf              = Tf
    self.local_num_steps = local_num_steps
    self.num_steps       = local_num_steps*self.mpi_comm.Get_size()

    self.dt       = Tf/self.num_steps
    self.t0_local = self.mpi_comm.Get_rank()*local_num_steps*self.dt
    self.tf_local = (self.mpi_comm.Get_rank()+1.0)*local_num_steps*self.dt

    self.x_final = None
    self.shape0 = None
  
    comm          = self.getMPIComm()
    my_rank       = self.getMPIComm().Get_rank()
    num_ranks     = self.getMPIComm().Get_size()

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

    self.start_layer,self.end_layer = self.getStepBounds()

    # this tracks if you are training or not,
    # this is intended to match the behavior of
    # the PyTorch Module class, note though torchbraid
    # uses evalNetwork and trainNetwork
    self.training = True

    self.enable_diagnostics = False

    self.first = True
    self.reverted = False
  # end __init__

  def initCore(self):
    cdef braid_Core core
    cdef PyBraid_Core py_fwd_core
    cdef braid_Core fwd_core
    cdef double tstart
    cdef double tstop
    cdef int ntime
    cdef MPI.Comm comm = self.getMPIComm()
    cdef int rank      = self.getMPIComm().Get_rank()
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
    if self.require_storage:
      braid_SetStorage(core,0)
    braid_SetMaxLevels(core, self.max_levels)
    braid_SetMaxIter(core, self.max_iters)
    braid_SetPrintLevel(core,self.print_level)
    braid_SetNRelax(core,-1,self.nrelax)
    braid_SetNRelax(core,0,0) # set F relax on fine grid
    braid_SetCFactor(core,-1,self.cfactor) # -1 implies chage on all levels
    braid_SetAbsTol(core,self.abs_tol)
    if self.skip_downcycle==0:
      braid_SetSkip(core,0)
    else:
      braid_SetSkip(core,1)

    #_braid_SetVerbosity(core,1)

    # store the c pointer
    py_core = PyBraid_Core()
    py_core.setCore(core)

    return py_core
  # end initCore

  def __del__(self):
    if self.py_core is not None:

      py_core = <PyBraid_Core> self.py_core
      core = py_core.getCore()

      # Destroy Braid Core C-Struct
      braid_Destroy(core) # this should be on

      self.py_core = None
    # end core

  def getNumSteps(self):
    """
    Get the total number of steps over all procesors.
    """
    return self.num_steps

  def diagnostics(self,enable):
    """
    This method tells torchbraid, to keep track of the feature vectors
    and parameters for eventual output. This is to help debug stability
    questions and other potential issues
    """

    self.enable_diagnostics = enable

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

  def getShape(self):
    return self.shape0 

  def initializeStates(self):
    try:
      t = 0.0
      for i in range(self.local_num_steps+1):
        t = self.t0_local + i*self.dt
        u_vec = self.getUVector(0,t)
        if u_vec!=None:
          self.initializeVector(t,u_vec)
    except:
      output_exception("{}:initializeStates: rank {}, t={}".format(self.prefix_str,self.getMPIComm().Get_rank(),t))
   
  # end initializeStates

  def finalRelax(self):
    """
    Force the application to do a final FC relaxtion sweep. This is useful for
    computing the gradient in the backpropagation or adjoint method.
    """
    cdef PyBraid_Core py_core = <PyBraid_Core> self.py_core
    cdef braid_Core core = py_core.getCore()
    braid_SetFinalFCRelax(core)

  def runBraid(self,x):
    cdef PyBraid_Core py_core = <PyBraid_Core> self.py_core
    cdef braid_Core core = py_core.getCore()
    try:
       py_core = <PyBraid_Core> self.py_core
       core = py_core.getCore()

       self.setInitial(x)
 
       # Run Braid
       if not self.first:
         self.initializeStates()
       self.first = False

       braid_Drive(core) # my_step -> App:eval -> resnet "basic block"

       self.printBraidStats()

       fin = self.getFinal()
       self.x0 = None
       self.x_final = None
    except:
      output_exception('runBraid')

    return fin

  def getBraidStats(self):
    cdef PyBraid_Core py_core = <PyBraid_Core> self.py_core
    cdef braid_Core core = py_core.getCore()

    cdef double resnorm 
    cdef int iter_cnt 
    cdef int niter = -1 # used for lookup

    braid_GetNumIter(core, &iter_cnt);

    niter = -1
    braid_GetRNorms(core, &niter, &resnorm);

    return iter_cnt,resnorm
  # end printBraidStats

  def printBraidStats(self):
    cdef PyBraid_Core py_core = <PyBraid_Core> self.py_core
    cdef braid_Core core = py_core.getCore()

    cdef double resnorm 
    cdef int iter_cnt 
    cdef int niter = -1 # used for lookup

    # no printing internally enabled
    if self.tb_print_level==0:
      return

    braid_GetNumIter(core, &iter_cnt);

    niter = -1
    braid_GetRNorms(core, &niter, &resnorm);

    my_rank       = self.getMPIComm().Get_rank()
    if my_rank==0:
      print('  -- \"%s\" %03d iters yields rnorm of %.6e' % (self.prefix_str,iter_cnt,resnorm))
  # end printBraidStats

  def getCore(self):
    return self.py_core    
 
  def setPrintLevel(self,print_level,tb_print=False):
    """
    Set the print level for this object. If tb_print (torchbraid print) is
    set to true this method sets the internal printing diagnostics. If it is
    false, the print level is passed along to xbraid.
    """

    if tb_print:
      # short circuit and set internal level
      self.tb_print_level = print_level
    else:
      # set (default) xbraid printing 
      self.print_level = print_level

      core = (<PyBraid_Core> self.py_core).getCore()
      braid_SetPrintLevel(core,self.print_level)

  def setNumRelax(self,relax,level=-1):
    self.nrelax = relax 

    core = (<PyBraid_Core> self.py_core).getCore()
    braid_SetNRelax(core,level,self.nrelax)

  def getMaxIters(self):
    return self.max_iters

  def setMaxIters(self,max_iters):
    self.max_iters = max_iters

    core = (<PyBraid_Core> self.py_core).getCore()
    braid_SetMaxIter(core, self.max_iters)

  def setFMG(self):
    core = (<PyBraid_Core> self.py_core).getCore()
    braid_SetFMG(core)

  def setCFactor(self,cfactor):
    self.cfactor = cfactor 

    core = (<PyBraid_Core> self.py_core).getCore()
    if isinstance(cfactor,dict):
      for level in sorted(cfactor.keys()):
        braid_SetCFactor(core,level,cfactor[level]) # -1 implies chage on all levels
    
    else: 
      braid_SetCFactor(core,-1,self.cfactor) # -1 implies chage on all levels

  def setSkipDowncycle(self,skip):
    if skip:
      self.skip_downcycle = 1 
    else:
      self.skip_downcycle = 0 

    core = (<PyBraid_Core> self.py_core).getCore()
    braid_SetSkip(core,self.skip_downcycle)

  def setRevertedRanks(self,reverted):
    self.reverted = reverted 
    core = (<PyBraid_Core> self.py_core).getCore()
    braid_SetRevertedRanks(core,reverted)
    self.start_layer,self.end_layer = self.getStepBounds()

  def getUVector(self,level,t):
    cdef braid_Core core = (<PyBraid_Core> self.py_core).getCore()
    cdef braid_BaseVector bv

    with self.timer("getUVector"): 
      
      index = self.getGlobalTimeIndex(t)
      _braid_UGetVectorRef(core, level,index,&bv)

      # this can be null, return that the vector was not found
      if <unsigned int>(bv)!=0:
        return <object> bv.userVector
      else:
        return None

  def getMPIComm(self):
    return self.mpi_comm

  def getGlobalTimeIndex(self,t):
    return round(t / self.dt)

  def setInitial(self,x0):
    cdef braid_Core core = (<PyBraid_Core> self.py_core).getCore()
    cdef braid_BaseVector bv 

    if not self.reverted and self.mpi_comm.Get_rank()!=0:
      return

    if self.reverted and self.mpi_comm.Get_rank()!=self.mpi_comm.Get_size()-1:
      return

    self.x0 = BraidVector(x0,0)

    # set the appropriate initial condition
    if core.warm_restart:
      _braid_UGetVectorRef(core, 0, 0, &bv);
      if not (bv is NULL):
        py_bv = <object> bv.userVector
        py_bv.replaceTensor(x0)

  def initializeVector(self,t,x):  
    pass

  def buildInit(self,t):
    try:
      if t>0:
        zeros = [torch.zeros(s) for s in self.shape0]
        x = BraidVector(tuple(zeros),0)
      else:
        x = BraidVector(self.x0.tensors(),0)
  
      # an inherited function to initialize the vector
      # here you would assign weights
      self.initializeVector(t,x)
    except:
      output_exception('runBraid')

    return x

  def access(self,t,u):
    if t==self.Tf:
      self.x_final = u.clone()

  def getFinal(self):
    if self.x_final==None:
      return None
      
    # assert the level
    assert(self.x_final.level()==0)
    x_final_tensors = self.x_final.tensors()

    return x_final_tensors

  def evalNetwork(self):
    self.training = False

  def trainNetwork(self):
    self.training = True

  def getStepBounds(self):
    cdef braid_Core core = (<PyBraid_Core> self.py_core).getCore()
    cdef int ilower
    cdef int iupper
    _braid_GetDistribution(core, &ilower,&iupper)

    return ilower,iupper
# end BraidApp
