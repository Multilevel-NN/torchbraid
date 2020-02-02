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

ctypedef PyObject _braid_App_struct 
ctypedef _braid_App_struct* braid_App

ctypedef PyObject* braid_Vector

# to supress a warning from numpy
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

class ODEBlock(nn.Module):
  def __init__(self,layer,dt):
    super(ODEBlock, self).__init__()

    self.dt = dt
    self.layer = layer

  def forward(self, x):
    return x + self.dt*self.layer(x)
# end ODEBlock

class Model(torch.nn.Module):

  def __init__(self,comm,layer_block,num_steps,Tf,max_levels=1,max_iters=10):
    super(Model,self).__init__()

    # optional parameters
    self.max_levels  = max_levels
    self.max_iters   = max_iters
    self.print_level = 2
    self.nrelax = 0
    self.cfactor = 2

    self.mpi_data = MPIData(comm)
    self.Tf = Tf
    self.local_num_steps = num_steps
    self.num_steps       = num_steps*self.mpi_data.getSize()

    self.dt = Tf/self.num_steps
    self.t0_local = self.mpi_data.getRank()*num_steps*self.dt
    self.tf_local = (self.mpi_data.getRank()+1.0)*num_steps*self.dt
  
    self.layer_models = [layer_block() for i in range(self.local_num_steps)]
    self.local_layers = torch.nn.Sequential(*self.layer_models)

    self.x_final = None
  # end __init__
 
  def setPrintLevel(self,print_level):
    self.print_level = print_level

  def setNumRelax(self,relax):
    self.nrelax = relax 

  def setCFactor(self,cfactor):
    self.cfactor = cfactor 

  def getMPIData(self):
    return self.mpi_data

  def forward(self,x):

    self.setInitial(x)
    py_core = self.initCore()

    cdef braid_Core core = <braid_Core> py_core
 
    # Set Braid options
    braid_SetMaxLevels(core, self.max_levels)
    braid_SetMaxIter(core, self.max_iters)
    braid_SetPrintLevel(core,self.print_level)
    braid_SetNRelax(core,-1,self.nrelax)
    braid_SetCFactor(core,-1,self.cfactor) # -1 implies chage on all levels
 
    # Run Braid
    braid_Drive(core)

#    # Destroy Braid Core C-Struct
# 
# This line causes now end of suffering, with essentially
# random failures on exit associated with a seg fault
# or incorrectly freed piece of memory.
#
# Clearly this is the wrong thing to do, but for now,
# lets work with it (See Issue#1)
#
#    braid_Destroy(core)

    f = self.getFinal()

    return f
  # end forward

  def getLayerIndex(self,t):
    return round((t-self.t0_local) / self.dt)

  def setInitial(self,x0):
    self.x0 = x0

  def buildInit(self,t):
    x = self.x0.clone()
    if t>0:
      x[:] = 0.0
    return x

  def eval(self,x,tstart,tstop):
    #print('evaluating = (%f,%f) - %d,%d' % (tstart,tstop,
    #                                        self.getLayerIndex(tstart),
    #                                        self.getLayerIndex(tstop)))
    with torch.no_grad(): 
      return x+self.dt*self.layer_models[self.getLayerIndex(tstart)](x)

  # This method copies the layerr parameters and can be used for verification
  def buildSequentialOnRoot(self):
    ode_layers    = [ODEBlock(l,self.dt) for l in self.layer_models]
    remote_layers = ode_layers
    build_seq_tag = 12         # this 
    comm          = self.mpi_data.getComm()
    my_rank       = self.mpi_data.getRank()
    num_ranks     = self.mpi_data.getSize()

    # short circuit for serial case
    if num_ranks==1:
      return torch.nn.Sequential(*remote_layers)

    if my_rank==0:
      for i in range(1,self.mpi_data.getSize()):
        remote_layers += comm.recv(source=i,tag=build_seq_tag)
      return torch.nn.Sequential(*remote_layers)
    else:
      comm.send(ode_layers,dest=0,tag=build_seq_tag)
      return None
  # end buildSequentialOnRoot

  def access(self,t,u):
    if t==self.Tf:
      self.x_final = u.clone()

  def getFinal(self):
    return self.x_final

  def initCore(self):
    cdef braid_Core core
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

    return <object> core
# end Model

# Other helper functions (mostly for testing)
#################################

# This frees a an initial vector
# using the `my_free` function. 
def freeVector(app,u):
  cdef braid_App c_app = <PyObject*>app
  cdef braid_Vector c_u = <PyObject*>u

  my_free(c_app,c_u)

# This builds a close of the initial vector
# using the `my_init` function called from 'c'
def cloneInitVector(app):
  cdef braid_App c_app = <PyObject*>app
  cdef braid_Vector v_vec
  my_init(c_app,0.0,&v_vec)
  return (<object> v_vec)

# This builds a close of the initial vector
# using the `my_clone` 
def cloneVector(app,x):
  cdef braid_App c_app = <PyObject*>app
  cdef braid_Vector c_x = <PyObject*>x
  cdef braid_Vector v 
  my_clone(c_app,c_x,&v)

  return <object> v

def addVector(app,alpha,x,beta,y):
  cdef braid_App c_app = <PyObject*>app
  cdef double dalpha = alpha
  cdef braid_Vector c_x = <PyObject*>x
  cdef double dbeta  = beta
  cdef braid_Vector c_y = <PyObject*>y

  my_sum(c_app,dalpha,c_x,dbeta,c_y)

def vectorNorm(app,x):
  cdef braid_App c_app = <PyObject*>app
  cdef braid_Vector c_x = <PyObject*>x
  cdef double [1] norm = [ 0.0 ]
  
  my_norm(c_app,c_x,norm)

  return norm[0]

def bufSize(app):
  cdef braid_App c_app = <PyObject*>app
  cdef int [1] sz = [0]
  cdef braid_BufferStatus status = NULL
  
  my_bufsize(c_app,sz,status)

  return sz[0]

def allocBuffer(app):
  cdef void * buffer = PyMem_Malloc(bufSize(app))
  return <object> buffer

def freeBuffer(app,obuffer):
  cdef void * buffer = <void*> obuffer
  PyMem_Free(buffer)

def pack(app,vec,obuffer):
  cdef braid_App c_app    = <PyObject*>app
  cdef braid_Vector c_vec = <PyObject*>vec
  cdef void * buffer = <void*> obuffer
  cdef braid_BufferStatus status = NULL

  cdef double * dbuffer = <double*> buffer

  my_bufpack(c_app, c_vec, buffer,status)

def unpack(app,obuffer):
  cdef braid_App c_app    = <PyObject*>app
  cdef braid_Vector c_vec    
  cdef void * buffer = <void*> obuffer
  cdef braid_BufferStatus status = NULL
  
  my_bufunpack(c_app,buffer,&c_vec,status)

  return <object> c_vec
