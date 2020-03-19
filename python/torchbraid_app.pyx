# cython: profile=True
# cython: linetrace=True

import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from cpython.ref cimport PyObject, Py_INCREF, Py_DECREF

from mpi4py import MPI
cimport mpi4py.MPI as MPI
cimport mpi4py.libmpi as libmpi

import pickle # we need this for building byte packs

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

include "./braid.pyx"
include "./torchbraid_callbacks.pyx"

#  a python level module
##########################################################

def braid_UGetVector(py_core, int level,int index):
  cdef braid_Core core = (<PyBraid_Core> py_core).getCore()
  cdef braid_BaseVector bv
  _braid_UGetVectorRef(core, level, index,&bv)

  return <object> bv.userVector

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

class BraidApp:

  def __init__(self,comm,layer_models,num_steps,Tf,max_levels,max_iters,
                    fwd_app=None):

    # optional parameters
    self.max_levels  = max_levels
    self.max_iters   = max_iters
    self.print_level = 2
    self.nrelax      = 0
    self.cfactor     = 2

    self.mpi_data        = MPIData(comm)
    self.Tf              = Tf
    self.local_num_steps = num_steps
    self.num_steps       = num_steps*self.mpi_data.getSize()

    self.dt       = Tf/self.num_steps
    self.t0_local = self.mpi_data.getRank()*num_steps*self.dt
    self.tf_local = (self.mpi_data.getRank()+1.0)*num_steps*self.dt
  
    self.layer_models = layer_models

    self.py_core = None
    self.x_final = None

    self.skip_downcycle = 0
    self.param_size     = 0

    self.use_adjoint = False
    self.fwd_app     = None
    if fwd_app!=None:
      # this is a adjoint/backprop application
      self.use_adjoint = True
      self.fwd_app     = fwd_app

    # build up the core
    self.py_core = self.initCore()
  # end __init__

  def __del__(self):
    if self.py_core!=None:
      py_core = <PyBraid_Core> self.py_core
      core = py_core.getCore()

      # Destroy Braid Core C-Struct
      # FIXME: braid_Destroy(core) # this should be on
    # end core

  def getCore(self):
    return self.py_core    
 
  def setPrintLevel(self,print_level):
    self.print_level = print_level

    core = (<PyBraid_Core> self.py_core).getCore()
    braid_SetPrintLevel(core,self.print_level)

  def setNumRelax(self,relax):
    self.nrelax = relax 

    core = (<PyBraid_Core> self.py_core).getCore()
    braid_SetNRelax(core,-1,self.nrelax)

  def setCFactor(self,cfactor):
    self.cfactor = cfactor 

    core = (<PyBraid_Core> self.py_core).getCore()
    braid_SetCFactor(core,-1,self.cfactor) # -1 implies chage on all levels

  def setSkipDowncycle(self,skip):
    self.skip_downcycle = skip

    core = (<PyBraid_Core> self.py_core).getCore()
    braid_SetSkip(core,self.skip_downcycle)

  def getMPIData(self):
    return self.mpi_data

  def run(self,x):

    self.setInitial(x)

    cdef PyBraid_Core py_core = <PyBraid_Core> self.py_core
    cdef braid_Core core = py_core.getCore()

    # Run Braid
    braid_Drive(core)

    f = self.getFinal()

    return f
  # end forward

  def getTimeStepIndex(self,t,tf,level):
    return round((t-self.t0_local) / self.dt)

  def getLayer(self,t,tf,level):
    return self.layer_models[self.getTimeStepIndex(t,tf,level)]

  def getPrimalIndex(self,t,tf,level):
    ts = round(tf / self.dt)
    return  self.num_steps - ts

  def setInitial(self,x0):
    self.x0 = BraidVector(x0,0)

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

  def eval(self,x,tstart,tstop):
    dt = tstop-tstart

    if not self.use_adjoint:
      with torch.no_grad(): 
        t_x = x.tensor()
        layer = self.getLayer(tstart,tstop,x.level())
        t_y = t_x+dt*layer(t_x)
        return BraidVector(t_y,x.level()) 
    else:
      finegrid = 0
      primal_index = self.getPrimalIndex(tstart,tstop,x.level())

      # get the primal vector from the forward app
      px = <object> braid_UGetVector(self.fwd_app.getCore(),finegrid,primal_index)
      t_px = px.tensor().clone()
      t_px.requires_grad = True

      # because we need to get the layer from the forward app, this
      # transformation finds the right layer for the forward app seen
      # from the backward apps perspective
      layer = self.fwd_app.getLayer(self.Tf-tstop,self.Tf-tstart,x.level())
 
      # enables gradient calculation 
      with torch.enable_grad():
        t_py = t_px+dt*layer(t_px)
 
      t_x = x.tensor()
      t_py.backward(t_x)

      return BraidVector(t_px.grad,x.level()) 

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
    braid_SetCFactor(core,-1,self.cfactor) # -1 implies chage on all levels
    braid_SetSkip(core,self.skip_downcycle)

    # setup adjoint specific stuff
    if self.use_adjoint:
      py_fwd_core = self.fwd_app.getCore()
      fwd_core = py_fwd_core.getCore()
      braid_SetStorage(fwd_core,0)

      # reverse ordering for adjoint/backprop
      braid_SetRevertedRanks(core,1)
       

    # store the c pointer
    py_core = PyBraid_Core()
    py_core.setCore(core)

    return py_core
  # end initCore

  def maxParameterSize(self):
    if self.param_size==0:
      # walk through the sublayers and figure
      # out the largeset size
      for lm in self.layer_models:
        local_size = len(pickle.dumps(lm))
        self.param_size = max(local_size,self.param_size)
    
    return self.param_size
  # end maxParameterSize

# end Model
