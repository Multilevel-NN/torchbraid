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

from typing import Union
from libc.stdio cimport FILE, stdout
from torchbraid.braid_vector import BraidVector

cimport mpi4py.MPI as MPI

include "./braid.pyx"
include "./torchbraid_callbacks.pyx"

#  a python level module
##########################################################

def output_exception(label):
  s = traceback.format_exc()
  print('\n**** Torchbraid Callbacks::{} Exception ****\n{}'.format(label,s))

class BraidApp:

  def __init__(self,prefix_str,comm,num_steps,Tf,max_levels,max_iters,
               spatial_ref_pair=None,user_mpi_buf=False,
               require_storage=False,abs_tol=1e-12):

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
    self.num_steps       = num_steps
    self.local_num_steps = int(num_steps/self.mpi_comm.Get_size())
    assert(self.local_num_steps*self.mpi_comm.Get_size()==self.num_steps)

    self.dt       = Tf/self.num_steps
    self.t0_local = self.mpi_comm.Get_rank()*self.local_num_steps*self.dt
    self.tf_local = (self.mpi_comm.Get_rank()+1.0)*self.local_num_steps*self.dt

    self.x_final = None
    self.shape0 = None

    self.buffer = []

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

    self.user_mpi_buf = user_mpi_buf
    # turn on user-allocated MPI buffers

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

    self.device = None
    self.use_cuda = False
  # end __init__

  def getNumSteps(self):
    """
    Get the total number of steps over all procesors.
    """
    return self.num_steps

  def setPrintLevel(self,print_level,tb_print=False):
    """
    Set the print level for this object. 

    If tb_print (torchbraid print) is
    set to true this method sets the internal printing diagnostics. If it is
    false, the print level is passed along to xbraid.

    Parameters
    ----------
   
    print_level : int
      Integer print level, higher is more output. 0 is no output

    tb_print : bool
      Set the braid print level if False, otherwise set torchbraids print level.
    """

    if tb_print:
      # short circuit and set internal level
      self.tb_print_level = print_level
    else:
      # set (default) xbraid printing 
      self.print_level = print_level

      core = (<PyBraid_Core> self.py_core).getCore()
      braid_SetPrintLevel(core,self.print_level)
      # Always turn off the braid.out.cycle file
      braid_SetFileIOLevel(core, 0)

  def getFeatureShapes(self,tidx : int,level : int) -> list:
    """
    Get the shape of the feature tensor at a time index and level.

    Using the global (independent of the number of processors), time index
    on a specified level in the time step hierarchy get the shape of the
    feature vector (for neural ODEs this is the solution to the ODE). 

    Parameters
    ----------

    tidx : int
      The global time index on the level

    level : int
      The level the time index is with respect to.

    Returns
    -------

    A list containing the shapes associated with the feature tensor at the
    specified level and time index.
    """

    return list(self.shape0)

  def getParameterShapes(self,tidx : int,level : int) -> list:
    """
    Get the shape of the weigts and biases to be communicated at a time index and level.

    Using the global (independent of the number of processors), time index
    on a specified level in the time step hierarchy get the shape of the
    weights and biases if they need to be communicated. Typically for forward propagation
    they do, but for backward propgation they don't and the default (return an empty list)
    can be used

    Parameters
    ----------

    tidx : int
      The global time index on the level

    level : int
      The level the time index is with respect to.

    Returns
    -------

    A list containing the shapes associated with the weight tensors at the
    specified level and time index that need to be communciated. This can
    be an empty list (and is by default). Typically this is non-empty for the
    forward pass, and an empty list for the backward pass.
    """
    return [] # empty size, no rank no size

  def getFineTimeIndex(self,tidx,level):
    """
    Compute the global time index on the fine level.    

    Using the global (independent of the number of processors), time index
    on a specified level in the time step hierarchy compute the global time
    index on the finest level. This uses the stored coarsening factors
    to compute the fine level index. 

    Parameters
    ----------

    tidx : int
      The global time index on the level

    level : int
      The level the time index is with respect to.
    """

    if isinstance(self.cfactor,int):
      return tidx * self.cfactor**level

    # the coarsening factor is different on each level
    assert(isinstance(self.cfactor,dict))

    # build a list containing levels from top to bottom
    cfactors = [self.cfactor[l] for l in range(level-1,-1,-1)]
    for cf in cfactors:
      tidx = cf * tidx

    return tidx


  def setCFactor(self,cfactor : Union[int, dict]):
    """
    Change the coarsening factor.

    Set the coarsening factor. If an integer then this 
    is set on all levels. However, if this is a dictionary
    then the convection follows the braid convection. So
    an cfactor[0] defines the coarsening rate going from
    level 0 to level 1.

    Parameters
    ----------

    cfactor : int | dict
      The coarsening factor(s) to be used.
    """

    self.cfactor = cfactor 

    core = (<PyBraid_Core> self.py_core).getCore()
    if isinstance(cfactor,dict):
      for level in sorted(cfactor.keys()):
        braid_SetCFactor(core,level,cfactor[level]) # -1 implies chage on all levels
    
    else: 
      braid_SetCFactor(core,-1,self.cfactor) # -1 implies chage on all levels

  def finalRelax(self):
    """
    Force the application to do a final FC relaxtion sweep. 

    This is useful for computing the gradient in the backpropagation or adjoint method.
    """
    cdef PyBraid_Core py_core = <PyBraid_Core> self.py_core
    cdef braid_Core core = py_core.getCore()
    braid_SetFinalFCRelax(core)
  
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
    cdef braid_PtFcnBufAlloc b_bufalloc = <braid_PtFcnBufAlloc> my_bufalloc
    cdef braid_PtFcnBufFree b_buffree = <braid_PtFcnBufFree> my_buffree

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
    
    if self.user_mpi_buf:
      braid_SetBufAllocFree(core, b_bufalloc, b_buffree)

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
    braid_SetAccessLevel(core,0)
    #braid_SetCRelaxWt(core, -1, 1.2)   # Turn on weighted relaxation, probably want to add command line argument
    braid_SetFileIOLevel(core, 0)       # Always turn off the braid.out.cycle file
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

  def setDevice(self,device):
    cdef braid_PtFcnBufAlloc b_bufalloc = <braid_PtFcnBufAlloc> my_bufalloc
    cdef braid_PtFcnBufFree b_buffree = <braid_PtFcnBufFree> my_buffree
    cdef PyBraid_Core py_core = <PyBraid_Core> self.py_core
    cdef braid_Core core = py_core.getCore()

    self.device = device

    self.use_cuda = False
    if torch.cuda.is_available():
      self.use_cuda = self.device.type=='cuda'

    if self.use_cuda:
      self.user_mpi_buf = True
      braid_SetBufAllocFree(core, b_bufalloc, b_buffree)

  def diagnostics(self,enable):
    """
    This method tells torchbraid, to keep track of the feature vectors
    and parameters for eventual output. This is to help debug stability
    questions and other potential issues
    """

    self.enable_diagnostics = enable

  def buildShapes(self,x):
    return x.size()

  def setShape(self,shape):
    # the shape to use if non-exists for taking advantage of allocations in braid
    if isinstance(shape,torch.Size):
      self.shape0 = [shape,]
    elif isinstance(shape,tuple):
      assert(False)
    else:
      self.shape0 = shape

  def getShape(self):
    return self.shape0

  def addBufferEntry(self, tensor):
    self.buffer.append(tensor)
    return self.buffer[-1].data_ptr()

  def getBuffer(self, addr):
    for i in range(len(self.buffer)):
      dataPtr = self.buffer[i].data_ptr()
      if dataPtr == addr:
        return self.buffer[i]

    raise Exception('Buffer not found')

  def removeBufferEntry(self, addr):
    self.buffer = [item for item in self.buffer if item.data_ptr() != addr]

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

  def testBraid(self, x):
    """
    Run some Braid Diagnostics
    """

    cdef PyBraid_Core py_core = <PyBraid_Core> self.py_core
    cdef braid_Core core = py_core.getCore()
    cdef MPI.Comm comm = self.getMPIComm()
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
    
    py_core = <PyBraid_Core> self.py_core
    core = py_core.getCore()

    self.setInitial(x)
 
    if not self.first:
        self.initializeStates()
    
    self.first = False

    # Other test functions possible.  See braid.pyx.
    #braid_TestBuf(<braid_App> self, comm.ob_mpi, stdout, 0.0,
    #              b_init, b_free, b_sum, b_norm, b_bufsize, b_bufpack, b_bufunpack) 

    #braid_TestCoarsenRefine(<braid_App> self, comm.ob_mpi, stdout, 0.0, 0.1, 0.2, 
    #                        b_init, b_access, b_free, b_clone, b_sum, b_norm, 
    #                        b_coarsen, b_refine)            

    #if self.spatial_mg:
    #    braid_TestAll( <braid_App> self, comm.ob_mpi, stdout, 0.0, 0.1, 0.2, 
    #                b_init, b_free, b_clone, b_sum, b_norm, b_bufsize, b_bufpack, b_bufunpack,
    #                b_coarsen, b_refine, NULL, b_step)    
    
    #else:
    #    braid_TestAll( <braid_App> self, comm.ob_mpi, stdout, 0.0, 0.1, 0.2, 
    #                b_init, b_free, b_clone, b_sum, b_norm, b_bufsize, b_bufpack, b_bufunpack,
    #                NULL,    NULL,   NULL,    b_step)    

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

       with self.timer("braid_Drive"):
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
 
  def setStorage(self, storage):
    core = (<PyBraid_Core> self.py_core).getCore()
    braid_SetStorage(core, storage)

  def setMinCoarse(self, mc):
    core = (<PyBraid_Core> self.py_core).getCore()
    braid_SetMinCoarse(core, mc)

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

  def setAbsTol(self,abs_tol):
    self.abs_tol = abs_tol

    core = (<PyBraid_Core> self.py_core).getCore()
    braid_SetAbsTol(core,self.abs_tol)

  def setFMG(self):
    core = (<PyBraid_Core> self.py_core).getCore()
    braid_SetFMG(core)

  def setCRelaxWt(self, CWt):
    core = (<PyBraid_Core> self.py_core).getCore()
    braid_SetCRelaxWt(core, -1, CWt)

  def setRelaxOnlyCG(self, flag):
    core = (<PyBraid_Core> self.py_core).getCore()
    braid_SetRelaxOnlyCG(core, flag)


  def setTimerFile(self, filestem):
    core = (<PyBraid_Core> self.py_core).getCore()
    braid_SetTimerFile(core, len(filestem), filestem.encode('utf-8'))

  def resetBraidTimer(self):
    core = (<PyBraid_Core> self.py_core).getCore()
    braid_ResetTimer(core)

  def setBraidTimers(self, flag):
    core = (<PyBraid_Core> self.py_core).getCore()
    braid_SetTimings(core, flag)

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

    self.x0 = BraidVector(x0)

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
        glb_idx = self.getGlobalTimeIndex(t)
        zeros = [torch.zeros(s,device=self.device) for s in self.getFeatureShapes(glb_idx,0)]
        x = BraidVector(tuple(zeros))
      else:
        x = BraidVector(self.x0.tensors())
  
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

  def getTimePoints(self):
    cdef braid_BaseVector bv 
    cdef braid_Core core = (<PyBraid_Core> self.py_core).getCore()


    
    times  = []
    values = []
    for i in range(core.grids[0].ilower,core.grids[0].iupper+1):
      _braid_UGetVectorRef(core, 0, i, &bv)
     
      times  += [core.grids[0].ta[i-core.grids[0].ilower]]
      values += [(<object> bv.userVector).clone()]

    return times,values
 

  def print_network(self, filename, state=True, parameters=True):
    '''
    Print the network to filename.  Can choose to print the state and/or parameters
    
    Filename format is:
      state :     fname.rank.timestep.tensor_index
      parameters: fname.rank.parameter_index.subparameter_index
    '''

    my_rank = self.getMPIComm().Get_rank()
    
    # Print network state
    with torch.no_grad():
      if state:
        try:
          t = 0.0
          for i in range(self.local_num_steps+1):
            t = self.t0_local + i*self.dt
            u_vec = self.getUVector(0,t) # returns a Braid uservector, which is what you want
            for j,ten in enumerate(u_vec.tensor_data_):
              fname = filename + ".state." + str(my_rank) + "." + "%06d"%i + "." + "%02d"%j
              np.savetxt(fname, ten.numpy().flatten())
        
        except:
          output_exception("Unable to print network state to file") 
    
    # Print network state
    with torch.no_grad():
      try:
        if parameters:
          for j,params in enumerate(self.parameters()): 
            for k,pp in enumerate(params):
              fname = filename + ".params." + str(my_rank) + "." + "%06d"%j + "." + "%06d"%k
              np.savetxt(fname, pp.numpy().flatten())
        
      except:
        output_exception("Unable to print network parameters to file") 


  def interp_network_state(self, coarse_app, cf):
    
    ''' 
    Interp the network state present in coarse_app into this app.  It is
    assumed that this app is a regular refinement (relative to coarsening
    factor cf) of coarse_app.  
    '''
    

    # Get braid_Core
    cdef braid_Core fine_core = (<PyBraid_Core> self.py_core).getCore()
    cdef braid_Core coarse_core = (<PyBraid_Core> coarse_app.py_core).getCore()
    
    # Get fine and coarse grid information.  Note, braid traditionally uses
    # values like ncpoints, clower, and cupper for these calcuations, but we
    # don't do that here, because we want to treat the sequential Braid and
    # MGRIT Braid modes the same.
    cdef fine_ilower = fine_core.grids[0].ilower    
    cdef fine_iupper = fine_core.grids[0].iupper    
    cdef coarse_ilower = coarse_core.grids[0].ilower    
    cdef coarse_iupper = coarse_core.grids[0].iupper    
    #print(fine_ilower, fine_iupper, coarse_ilower, coarse_iupper)

    cdef braid_BaseVector fine_bv, coarse_bv

    # The number of C points in fine app (self) must match the number of points in the coarse_app
    fine_indices = np.arange(fine_ilower, fine_iupper+1)
    fine_indices = fine_indices[ (fine_indices % cf) == 0 ] # picks off the Cpts 
    coarse_indices = np.arange(coarse_ilower, coarse_iupper+1)
    #print(fine_indices, coarse_indices)
    assert( fine_indices.shape[0] == coarse_indices.shape[0])

    ##
    # Note, this is only designed to work in serial.  When moving to parallel,
    # we'll have to send messages and use GetProc
    ## 

    with torch.no_grad():
      for fine_pt, coarse_pt in zip(fine_indices, coarse_indices):
        
        # Get the coarse vector
        _braid_UGetVectorRef(coarse_core, 0, coarse_pt, &coarse_bv)
        assert(<unsigned int>(coarse_bv)!=0)
        py_coarse_bv = <object> coarse_bv.userVector
      
        # Do piece-wise constant to fill in fine-grid, F- and C-points
        for k in range(0,cf):

          # Get the fine vector, but if C-point make sure that pointer is non-Null
          _braid_UGetVectorRef(fine_core, 0, fine_pt+k, &fine_bv)
          if k == 0:
            assert(<unsigned int>(fine_bv)!=0) 
          
          # Copy coarse to fine at C-pt
          # F-points may or may not be stored.  Only copy if stored
          if (<unsigned int>(fine_bv) != 0):
            py_fine_bv = <object> fine_bv.userVector
            py_coarse_bv_clone = py_coarse_bv.clone() 
            py_fine_bv.replaceTensor(py_coarse_bv_clone.tensors())
            ##tensor_clone = tuple( [ ten.detach().clone() for ten in py_coarse_bv.tensor_data_])
            ##py_fine_bv.replaceTensor(tensor_clone)
        

  def inject_network_state(self, fine_app, cf):
    
    ''' 
    Inject the network state present in fine_app into this app.  It is
    assumed that fine_app is a regular refinement (relative to coarsening factor cf) 
    of this app.  
    '''
    

    # Get braid_Core
    cdef braid_Core fine_core = (<PyBraid_Core> fine_app.py_core).getCore()
    cdef braid_Core coarse_core = (<PyBraid_Core> self.py_core).getCore()
    
    # Get fine and coarse grid information.  Note, braid traditionally uses
    # values like ncpoints, clower, and cupper for these calcuations, but we
    # don't do that here, because we want to treat the sequential Braid and
    # MGRIT Braid modes the same.
    cdef fine_ilower = fine_core.grids[0].ilower    
    cdef fine_iupper = fine_core.grids[0].iupper    
    cdef coarse_ilower = coarse_core.grids[0].ilower    
    cdef coarse_iupper = coarse_core.grids[0].iupper    
    #print(fine_ilower, fine_iupper, coarse_ilower, coarse_iupper)

    cdef braid_BaseVector fine_bv, coarse_bv

    
    #cdef braid_BaseVector u0 = fine_core.grids[0].ua[0]
    #cdef braid_BaseVector u1 = fine_core.grids[0].ua[1]
    #cdef braid_BaseVector u2 = fine_core.grids[0].ua[2]
    #cdef braid_BaseVector uLast = fine_core.grids[0].ulast
    #print("EE 0:  " + str( <unsigned int>(u0) ))
    #print("EE 1:  " + str( <unsigned int>(u1) ))
    #print("EE 2:  " + str( <unsigned int>(u2) ))
    #print("EE L:  " + str( <unsigned int>(uLast) ))


    # The number of C points in fine_app must match the number of points in the coarse app (self)
    fine_indices = np.arange(fine_ilower, fine_iupper+1)
    fine_indices = fine_indices[ (fine_indices % cf) == 0 ] # picks off the Cpts 
    coarse_indices = np.arange(coarse_ilower, coarse_iupper+1)
    #print(fine_indices, coarse_indices)
    assert( fine_indices.shape[0] == coarse_indices.shape[0])

    with torch.no_grad():
      for fine_pt, coarse_pt in zip(fine_indices, coarse_indices):
        
        # Get the fine vector
        _braid_UGetVectorRef(fine_core, 0, fine_pt, &fine_bv)
        assert(<unsigned int>(fine_bv)!=0)
      
        # Get the coarse vector
        _braid_UGetVectorRef(coarse_core, 0, coarse_pt, &coarse_bv)
        assert(<unsigned int>(coarse_bv)!=0)
      
        # Copy fine to coarse  
        py_coarse_bv = <object> coarse_bv.userVector
        py_fine_bv = <object> fine_bv.userVector
        py_fine_bv_clone = py_fine_bv.clone() 
        py_coarse_bv.replaceTensor(py_fine_bv_clone.tensors())
        ##tensor_clone = tuple( [ ten.detach().clone() for ten in py_fine_bv.tensor_data_])
        ##py_coarse_bv.replaceTensor(tensor_clone)


  def GetProc(self, app, idx):
    ''' Helper function that returns the processor that owns time-point idx '''
    cdef int proc
    cdef braid_Core core = (<PyBraid_Core> app.py_core).getCore()
    _braid_GetProc(core, 0, idx, &proc)
    return proc

  def get_my_Cpoints(self, fine_ilower, fine_iupper, cf):
    ''' Helper function to returns the Cpoints present in this range of points '''
    clower = np.ceil(fine_ilower / cf) * cf
    cupper = int(fine_iupper / cf) * cf
    return np.arange(clower, cupper + 1, cf, dtype=int)


  def parallel_injection_interp_params(self, model_fine, model_coarse, cf=2, grad=False):
    
    ''' 
    Interpolate the model parameters according to coarsening-factor in time cf.

    Do this in parallel, assuming that the parameters are layed out in parallel
    the same way that the Braid layers are distributed
    
    Return a list of the interpolated model parameters.  Always do a deep copy.

    If grad is True, return the network gradient instead
    
    Note: The placement of this function is a bit odd.  We interpolate MGOpt
    solver parameters by calling a function that resides inside the forward
    (backward) BraidApp inside of MGOpt object. This was done because
    - This function must be on the .pyx level to access _braid functionality
    - This function will eventually (hopefully) also interpolate Braid state
      vectors, and then it will make sense to use the same function and MPI
      messages to do that.

    '''
    # See
    # https://stackoverflow.com/questions/383565/how-to-iterate-over-a-list-repeating-each-element-in-python
    def duplicate(iterable,n):
      """A generator that repeats each entry n times"""
      for item in iterable:
        first = True
        for _ in range(n):
          yield item,first
          first = False
    
    ##
    # Get foward apps 
    fine_fwd_app = model_fine.parallel_nn.fwd_app
    coarse_fwd_app = model_coarse.parallel_nn.fwd_app
    
    ##
    # Set up comm and return value interp_params
    comm = fine_fwd_app.mpi_comm
    my_rank = comm.Get_rank()
    num_ranks = comm.Get_size()
    open_params = None
    close_params = None
    send_requests = []
    recv_requests = []
    
    ##
    # Get braid_Core's, and ilower and iupper points on local time-grid indices
    cdef braid_Core fine_fwd_core = (<PyBraid_Core> fine_fwd_app.py_core).getCore()
    cdef braid_Core coarse_fwd_core = (<PyBraid_Core> coarse_fwd_app.py_core).getCore()
    cdef coarse_ilower = coarse_fwd_core.grids[0].ilower    # index of lowest layers owned by proc
    cdef coarse_iupper = coarse_fwd_core.grids[0].iupper    # index of highest layers owned by proc (inclusive)
    cdef coarse_gupper = coarse_fwd_core.grids[0].gupper    # global upper index of layers
    cdef fine_ilower = fine_fwd_core.grids[0].ilower
    cdef fine_iupper = fine_fwd_core.grids[0].iupper
    cdef fine_gupper = fine_fwd_core.grids[0].gupper
    cdef int tag

    ##
    # Check that your layer parallel torchbraid model is the same length as expected by Braid
    num_layer_parallel = 0
    with torch.no_grad():
      for child in model_coarse.children():
        name = str(type(child))
        name = name[ name.rfind('.')+1 : name.rfind('\'') ]
        if name == 'LayerParallel':
          for lp_child in child.layer_models:
            num_layer_parallel = num_layer_parallel + 1
    ##
    # On last _active_ processor, num_layer_parallel is one less than the number of layers
    if my_rank == self.GetProc(coarse_fwd_app, coarse_gupper): 
      num_layer_parallel = num_layer_parallel + 1
    if (num_layer_parallel != (coarse_iupper - coarse_ilower + 1) ):
      output_exception("parallel_injection_interp_params:  number of LayerParallel layers " + str(num_layer_parallel) +\
                       " not what was expected based on ilower and iupper, " + str(coarse_iupper - coarse_ilower + 1) )
    
    ##
    # Loop 1: loop over all children, MPI sending the interpolated layer-parallel weights 
    with torch.no_grad():
      for child in model_coarse.children():
        name = str(type(child))
        name = name[ name.rfind('.')+1 : name.rfind('\'') ]
        
        # handle layer parallel modules differently 
        if name == 'LayerParallel':

          fidx = 0
          # loop over each layer-parallel layer
          # note, that duplicate does the piece-wise interpolation
          for (lp_child, lp_f) in duplicate(child.layer_models, cf):
            lp_params = []
            for param in lp_child.parameters():
              if grad: lp_params.append(param.grad.clone().detach())
              else:    lp_params.append(param.clone().detach())
            # end for param loop
            
            # MPI tag is the LayerParallel index (on fine grid) of this time point
            tag = coarse_ilower*cf + fidx
            proc = self.GetProc(fine_fwd_app, tag)
            req = comm.isend(lp_params, dest=proc, tag=tag)  
            send_requests.append( req )
            
            fidx = fidx + 1
          # end for lp_child loop
                 
        else:
          
          # Do simple injection for the opening and closing layers.  
          lp_params=[]
          for param in child.parameters():
            if grad: lp_params.append(param.grad.clone().detach())    # Note the clone, i.e., deep copy
            else:    lp_params.append(param.clone().detach())
          
          if name == 'CloseLayer':
            close_params = lp_params
          elif name == 'OpenLayer':
            open_params = lp_params
          else:
            output_exception('Layer type needs to be OpenLayer, CloseLayer, or LayerParallel')

      # end for child loop
    
      ##
      # Begin construction of output.
      # Rank 0:    parameter order is LayerParallel, OpenLayer, CloseLayer
      # Rank k>0:  owns a chunk of LayerParallel
      interp_params = []
      
      ##
      # Loop over fine model, receiving all of your weights at every layer (F and C)
      # ==> Note that the global last time-point doesn't have a corresponding layer to receive 
      last_point = fine_iupper + 1
      if (fine_iupper == fine_gupper):
        last_point = fine_iupper
      for point in np.arange(fine_ilower, last_point):
      
        # Processor to receive from (owns Cpt to the left of "point")
        proc = self.GetProc(coarse_fwd_app, int(point/cf))
        # Receive params (rely on pickle to convert data types (list and torch.nn.Parameter)
        req = comm.irecv(source=proc, tag=point)
        recv_requests.append( req )

      for req in recv_requests:
        # Store params in order from layer 0 to layer gupper
        lp_params = req.wait()
        interp_params = interp_params + lp_params
      
      ##
      # If processor 0, interpolate OpenLayer and CloseLayer
      # Note: interpolation is just a "deep" copy of open_params
      if my_rank == 0:
        if open_params == None:
          output_exception('OpenLayer not found on rank 0')
        #
        interp_params = interp_params + open_params
        
        if close_params == None:
          output_exception('CloseLayer not found on rank 0')
        #
        interp_params = interp_params + close_params
      
      ##
      # Finish all sends
      for req in send_requests:
        req.wait()

    ##
    # If spatial coarsening is ever desired, an interpolation function could be
    # applied here to interp_params
    #
    # If/when we incorporate state interpolation here, we could write the
    # state and params directly into model_fine 
    
    return interp_params
  
  

  def parallel_injection_restrict_params(self, model_fine, model_coarse, cf=2, grad=False):
    
    ''' 
    Restrict the model parameters according to coarsening-factor in time cf.

    Do this in parallel, assuming that the parameters are layed out in parallel
    the same way that the Braid layers are distributed
    
    Return a list of the restricted model parameters.  Always do a deep copy.

    If grad is True, return the network gradient instead
    
    Note: The placement of this function is a bit odd.  See above discussion at
    start of parallel_injection_interp_params.  
    '''
    # See
    # https://stackoverflow.com/questions/383565/how-to-iterate-over-a-list-repeating-each-element-in-python
    def duplicate(iterable,n):
      """A generator that repeats each entry n times"""
      for item in iterable:
        first = True
        for _ in range(n):
          yield item,first
          first = False
    
    ##
    # Get foward apps 
    fine_fwd_app = model_fine.parallel_nn.fwd_app
    coarse_fwd_app = model_coarse.parallel_nn.fwd_app
    
    ##
    # Set up comm and return value interp_params
    comm = fine_fwd_app.mpi_comm
    my_rank = comm.Get_rank()
    num_ranks = comm.Get_size()
    open_params = None
    close_params = None
    send_requests = []
    recv_requests = []
    
    ##
    # Get braid_Core's, and ilower and iupper points on local time-grid indices
    cdef braid_Core fine_fwd_core = (<PyBraid_Core> fine_fwd_app.py_core).getCore()
    cdef braid_Core coarse_fwd_core = (<PyBraid_Core> coarse_fwd_app.py_core).getCore()
    cdef coarse_ilower = coarse_fwd_core.grids[0].ilower    # index of lowest layers owned by proc
    cdef coarse_iupper = coarse_fwd_core.grids[0].iupper    # index of highest layers owned by proc (inclusive)
    cdef coarse_gupper = coarse_fwd_core.grids[0].gupper    # global upper index of layers
    cdef fine_ilower = fine_fwd_core.grids[0].ilower
    cdef fine_iupper = fine_fwd_core.grids[0].iupper
    cdef fine_gupper = fine_fwd_core.grids[0].gupper
    cdef int tag

    ##
    # Check that your layer parallel torchbraid model is the same length as expected by Braid
    num_layer_parallel = 0
    with torch.no_grad():
      for child in model_fine.children():
        name = str(type(child))
        name = name[ name.rfind('.')+1 : name.rfind('\'') ]
        if name == 'LayerParallel':
          for lp_child in child.layer_models:
            num_layer_parallel = num_layer_parallel + 1
    ##
    # On last _active_ processor, num_layer_parallel is one less than the number of layers
    if my_rank == self.GetProc(fine_fwd_app, fine_gupper): 
      num_layer_parallel = num_layer_parallel + 1
    if (num_layer_parallel != (fine_iupper - fine_ilower + 1) ):
      output_exception("parallel_injection_restrict_params: number of LayerParallel layers " + str(num_layer_parallel) + " not what was expected based on ilower and iupper, " + str(fine_iupper - fine_ilower + 1) )
    
    ##
    # Loop 1: loop over all children, MPI sending the restricted layer-parallel weights 
    with torch.no_grad():
      for child in model_fine.children():
        name = str(type(child))
        name = name[ name.rfind('.')+1 : name.rfind('\'') ]
        
        # handle layer parallel modules differently 
        if name == 'LayerParallel':

          fidx = -1
          # loop over each layer-parallel layer
          for lp_child in child.layer_models: 
            # Only do restriction if C-point
            fidx = fidx + 1
            if ((fine_ilower + fidx) % cf) == 0:
              lp_params = []
              for param in lp_child.parameters():
                if grad: lp_params.append(param.grad.clone().detach())
                else:    lp_params.append(param.clone().detach())
              # end for param loop
              
              #MPI tag is the LayerParallel index (on coarse grid) of this time point
              tag = int( (fine_ilower + fidx) / cf )
              proc = self.GetProc(coarse_fwd_app, tag)
              req = comm.isend(lp_params, dest=proc, tag=tag)  
              send_requests.append( req )
            
          # end for lp_child loop
                 
        else:
          
          # Do simple injection for the opening and closing layers.  
          lp_params=[]
          for param in child.parameters():
            if grad: lp_params.append(param.grad.clone().detach())    # Note the clone, i.e., deep copy
            else:    lp_params.append(param.clone().detach())
          
          if name == 'CloseLayer':
            close_params = lp_params
          elif name == 'OpenLayer':
            open_params = lp_params
          else:
            output_exception('Layer type needs to be OpenLayer, CloseLayer, or LayerParallel')

      # end for child loop
    
      ##
      # Begin construction of output.
      # Rank 0:    parameter order is LayerParallel, OpenLayer, CloseLayer
      # Rank k>0:  owns a chunk of LayerParallel
      restrict_params = []
      
      ##
      # Loop over coarse model, receiving all of your weights at every layer (F and C)
      # ==> Note that the global last time-point doesn't have a corresponding layer to receive 
      last_point = coarse_iupper + 1
      if (coarse_iupper == coarse_gupper):
        last_point = coarse_iupper
      for point in np.arange(coarse_ilower, last_point):
        # Processor to receive from
        proc = self.GetProc(fine_fwd_app, int(point*cf))
        # Receive params (rely on pickle to convert data types (list and torch.nn.Parameter)
        req = comm.irecv(source=proc, tag=point)
        recv_requests.append( req )

      for req in recv_requests:
        # Store params in order from layer 0 to layer gupper
        lp_params = req.wait()
        restrict_params = restrict_params + lp_params
      
      ##
      # If processor 0, interpolate OpenLayer and CloseLayer
      # Note: interpolation is just a "deep" copy of open_params
      if my_rank == 0:
        if open_params == None:
          output_exception('OpenLayer not found on rank 0')
        #
        restrict_params = restrict_params + open_params
        
        if close_params == None:
          output_exception('CloseLayer not found on rank 0')
        #
        restrict_params = restrict_params + close_params
      
      ##
      # Finish all sends
      for req in send_requests:
        req.wait()
      
    ##
    # If spatial coarsening is ever desired, an interpolation function could be
    # applied here to restrict_params
    #
    # If/when we incorporate state interpolation, we could write the
    # state and params directly into model_fine instead of returning
    
    return restrict_params
# end BraidApp
