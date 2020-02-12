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

class BraidVector:
  def __init__(self,tensor,level):
    self.tensor_ = tensor 
    self.level_  = level

  def tensor(self):
    return self.tensor_

  def level(self):
    return self.level_
  
  def clone(self):
    cl = BraidVector(self.tensor().clone(),self.level())
    return cl

ctypedef PyObject _braid_Vector_struct
ctypedef _braid_Vector_struct *braid_Vector

##
# Define your Python Braid Vector

# to supress a warning from numpy
cdef extern from *:
  """
  #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
  """

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

  def __init__(self,comm,layer_block,num_steps,Tf,max_levels=1,max_iters=10,
                    coarsen=None,
                    refine=None):
    super(Model,self).__init__()

    # optional parameters
    self.max_levels  = max_levels
    self.max_iters   = max_iters
    self.print_level = 2
    self.nrelax = 0
    self.cfactor = 2

    self.mpi_data = MPIData(comm)
    self.Tf = Tf
    self.local_num_steps = dict()
    self.local_num_steps[0] = num_steps
    self.num_steps = dict()
    self.num_steps[0] = num_steps*self.mpi_data.getSize()

    self.dt = Tf/self.num_steps[0]
    self.t0_local = self.mpi_data.getRank()*num_steps*self.dt
    self.tf_local = (self.mpi_data.getRank()+1.0)*num_steps*self.dt
  
    self.layer_block = layer_block
    self.layer_models = dict()
    self.layer_models[0] = [layer_block() for i in range(self.local_num_steps[0])]
    self.local_layers = torch.nn.Sequential(*self.layer_models[0])

    self.py_core = None
    self.x_final = None

    if coarsen==None or refine==None:
      assert(coarsen==refine) # both should be None
      self.refinement_on = False
    else:
      self.refinement_on = True

    self.coarsen = coarsen
    self.refine  = refine
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

    if self.py_core==None:
      self.py_core = self.initCore()

    cdef PyBraid_Core py_core = <PyBraid_Core> self.py_core
    cdef braid_Core core = py_core.getCore()

    # Run Braid
    braid_Drive(core)

    # Destroy Braid Core C-Struct
    braid_Destroy(core)

    f = self.getFinal()

    return f
  # end forward

  def getLayer(self,t,level):
    return self.layer_models[0][round((t-self.t0_local) / self.dt)]

  def setInitial(self,x0):
    self.x0 = BraidVector(x0,0)

  def buildInit(self,t):
    x = self.x0.clone()
    if t>0:
      t_x = x.tensor()
      t_x[:] = 0.0
    return x

  def eval(self,x,tstart,tstop):
    with torch.no_grad(): 
      t_x = x.tensor()
      layer = self.getLayer(tstart,x.level())
      t_y = t_x+self.dt*layer(t_x)
      return BraidVector(t_y,x.level()) 

  # This method copies the layerr parameters and can be used for verification
  def buildSequentialOnRoot(self):
    ode_layers    = [ODEBlock(l,self.dt) for l in self.layer_models[0]]
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
    if self.x_final==None:
      return None

    # asse the level
    assert(self.x_final.level()==0)
    return self.x_final.tensor()

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
    cdef braid_PtFcnSCoarsen b_coarsen = <braid_PtFcnSCoarsen> my_coarsen
    cdef braid_PtFcnSRefine  b_refine  = <braid_PtFcnSRefine> my_refine

    ntime = self.num_steps[0]
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

    if self.refinement_on:
      braid_SetSpatialCoarsen(core,b_coarsen)
      braid_SetSpatialRefine(core,b_refine)
    # end if refinement_on

    # Set Braid options
    braid_SetMaxLevels(core, self.max_levels)
    braid_SetMaxIter(core, self.max_iters)
    braid_SetPrintLevel(core,self.print_level)
    braid_SetNRelax(core,-1,self.nrelax)
    braid_SetCFactor(core,-1,self.cfactor) # -1 implies chage on all levels

    # store the c pointer
    py_core = PyBraid_Core()
    py_core.setCore(core)

    return py_core
# end Model

# Other helper functions (mostly for testing)
#################################

# This frees a an initial vector
# using the `my_free` function. 
def freeVector(app,u):
  cdef braid_App c_app = <PyObject*>app
  cdef braid_Vector c_u = <braid_Vector> u

  my_free(c_app,c_u)

# This builds a close of the initial vector
# using the `my_init` function called from 'c'
def cloneInitVector(app):
  cdef braid_App c_app = <PyObject*>app
  cdef braid_Vector v_vec
  my_init(c_app,0.0,&v_vec)
  return (<object> v_vec).tensor()

# This builds a close of the initial vector
# using the `my_clone` 
def cloneVector(app,x):
  b_vec = BraidVector(x,0)

  cdef braid_App c_app = <PyObject*>app
  cdef braid_Vector c_x = <braid_Vector> b_vec
  cdef braid_Vector v 
  my_clone(c_app,c_x,&v)

  return (<object> v).tensor()

def addVector(app,alpha,ten_x,beta,ten_y):
  x = BraidVector(ten_x,0)
  y = BraidVector(ten_y,0)

  cdef braid_App c_app = <PyObject*>app
  cdef double dalpha = alpha
  cdef braid_Vector c_x = <braid_Vector>x
  cdef double dbeta  = beta
  cdef braid_Vector c_y = <braid_Vector>y

  my_sum(c_app,dalpha,c_x,dbeta,c_y)

def vectorNorm(app,ten_x):
  x = BraidVector(ten_x,0)

  cdef braid_App c_app = <PyObject*>app
  cdef braid_Vector c_x = <braid_Vector>x
  cdef double [1] norm = [ 0.0 ]
  
  my_norm(c_app,c_x,norm)

  return norm[0]

def bufSize(app):
  cdef braid_App c_app = <PyObject*>app
  cdef int [1] sz = [0]
  cdef braid_BufferStatus status = NULL
  
  my_bufsize(c_app,sz,status)

  # subtract the int size (for testing purposes)
  return sz[0] - sizeof(int)

def allocBuffer(app):
  cdef void * buffer = PyMem_Malloc(bufSize(app))
  return <object> buffer

def freeBuffer(app,obuffer):
  cdef void * buffer = <void*> obuffer
  PyMem_Free(buffer)

def pack(app,ten_vec,obuffer,level):
  vec = BraidVector(ten_vec,level)

  cdef braid_App c_app    = <PyObject*>app
  cdef braid_Vector c_vec = <braid_Vector> vec
  cdef braid_BufferStatus status = NULL
  cdef void * buffer = <void*> obuffer

  my_bufpack(c_app, c_vec, buffer,status)

def unpack(app,obuffer):
  cdef braid_App c_app    = <PyObject*>app
  cdef braid_Vector c_vec    
  cdef void * buffer = <void*> obuffer
  cdef braid_BufferStatus status = NULL
  
  my_bufunpack(c_app,buffer,&c_vec,status)

  vec = <object> c_vec
  return (vec.tensor(),vec.level())
