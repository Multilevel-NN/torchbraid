import torch
import numpy as np
from collections import OrderedDict

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from cpython.ref cimport PyObject, Py_INCREF, Py_DECREF

from mpi4py import MPI
cimport mpi4py.MPI as MPI
cimport mpi4py.libmpi as libmpi

ctypedef PyObject* braid_App
ctypedef PyObject* braid_Vector

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

  def getRank(self):
    return self.size
# helper class for the MPI communicator

class Model(torch.nn.Module):

  def __init__(self,comm,layer_block,num_steps,Tf,max_levels=1,max_iters=10):
    super(Model,self).__init__()

    # optional parameters
    self.max_levels = max_levels
    self.max_iters  = max_iters

    self.mpi_data = MPIData(comm)
    self.Tf = Tf
    self.num_steps = num_steps
  
    self.layer_models = [layer_block() for i in range(self.num_steps)]
    self.local_layers = torch.nn.Sequential(*self.layer_models)
  # end __init__

  def __del__(self):
    pass

  def forward(self,x):

    self.setInitial(x)
    py_core = self.initCore()

    cdef braid_Core core = <braid_Core> py_core
 
    # Set Braid options
    braid_SetMaxLevels(core, self.max_levels)
    braid_SetMaxIter(core, self.max_iters)
 
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
    dt = self.Tf/self.num_steps
    return round(t / dt)

  def setInitial(self,x0):
    self.x0 = x0

  def buildInit(self,t):
    x = self.x0.clone()
    return x

  def eval(self,x,tstart,tstop):
    #print('evaluating = (%f,%f) - %d,%d' % (tstart,tstop,
    #                                        self.getLayerIndex(tstart),
    #                                        self.getLayerIndex(tstop)))
    with torch.no_grad(): 
      dt = tstop-tstart
      return x+dt*self.layer_models[self.getLayerIndex(tstart)](x)

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

    ntime = self.num_steps
    tstart = 0.0
    tstop = self.Tf

    braid_Init(comm.ob_mpi, comm.ob_mpi, 
               tstart, tstop, ntime, 
               <braid_App> self, my_step,
               my_init, my_clone, my_free, my_sum, my_norm, my_access, my_bufsize,
               my_bufpack, my_bufunpack, 
               &core)

    return <object> core
# end Model

# Other helper functions (mostly for testing)
#################################

# This frees a an initial vector
# using the `my_free` function. 
def freeVector(app,u):
  cdef braid_App c_app = <PyObject*>app
  cdef braid_Vector c_u = <PyObject*>u;

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
  cdef braid_BufferStatus status
  
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
  cdef braid_BufferStatus status 

  cdef double * dbuffer = <double*> buffer

  my_bufpack(c_app, c_vec, buffer,status)

def unpack(app,obuffer):
  cdef braid_App c_app    = <PyObject*>app
  cdef braid_Vector c_vec    
  cdef void * buffer = <void*> obuffer
  cdef braid_BufferStatus status 
  
  my_bufunpack(c_app,buffer,&c_vec,status)

  return <object> c_vec
