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

from torchbraid_function import BraidFunction

##
# Define your Python Braid Vector

# to supress a warning from numpy
cdef extern from *:
  """
  #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
  """

include "./torchbraid_app.pyx"

#  a python level module
##########################################################

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
    global_steps = num_steps*comm.Get_size()

    self.dt = Tf/global_steps
  
    self.layer_block = layer_block
    self.layer_models = [layer_block() for i in range(num_steps)]
    self.local_layers = torch.nn.Sequential(*self.layer_models)

    self.fwd_app = BraidApp(comm,self.layer_models,num_steps,Tf,max_levels,max_iters)
    self.bwd_app = BraidApp(comm,self.layer_models,num_steps,Tf,max_levels,max_iters,self.fwd_app)

    self.param_size = 0
  # end __init__

  def setPrintLevel(self,print_level):
    self.fwd_app.setPrintLevel(print_level)
    self.bwd_app.setPrintLevel(print_level)

  def setNumRelax(self,relax):
    self.fwd_app.setNumRelax(relax)
    self.bwd_app.setNumRelax(relax)

  def setCFactor(self,cfactor):
    self.fwd_app.setCFactor(cfactor)
    self.bwd_app.setCFactor(cfactor)

  def setSkipDowncycle(self,skip):
    self.fwd_app.setSkipDowncycle(skip)
    self.bwd_app.setSkipDowncycle(skip)

  def getMPIData(self):
    return self.fwd_app.getMPIData()

  def forward(self,x):
    # we are doing this to take adavtage of
    # pytorch's autograd which functions "naturally"
    # with the torch.autograd.function
    params = list(self.parameters())
    return BraidFunction.apply(x,params,self.fwd_app,self.bwd_app) 
  # end forward

  def setInitial(self,x0):
    self.x0 = BraidVector(x0,0)

  def buildInit(self,t):
    x = self.x0.clone()
    if t>0:
      t_x = x.tensor()
      t_x[:] = 0.0
    return x

  # This method copies the layerr parameters and can be used for verification
  def buildSequentialOnRoot(self):
    ode_layers    = [ODEBlock(l,self.dt) for l in self.layer_models]
    remote_layers = ode_layers
    build_seq_tag = 12         # this 
    comm          = self.getMPIData().getComm()
    my_rank       = self.getMPIData().getRank()
    num_ranks     = self.getMPIData().getSize()

    # short circuit for serial case
    if num_ranks==1:
      return torch.nn.Sequential(*remote_layers)

    if my_rank==0:
      for i in range(1,self.getMPIData().getSize()):
        remote_layers += comm.recv(source=i,tag=build_seq_tag)
      return torch.nn.Sequential(*remote_layers)
    else:
      comm.send(ode_layers,dest=0,tag=build_seq_tag)
      return None
  # end buildSequentialOnRoot

  def getFinal(self):
    return  self.fwd_app.getFinal()

  def getFinalOnRoot(self):
    build_seq_tag = 99        # this 
    comm          = self.getMPIData().getComm()
    my_rank       = self.getMPIData().getRank()
    num_ranks     = self.getMPIData().getSize()

    # short circuit for serial case
    if num_ranks==1:
      return self.getFinal()

    # send the output of the last layer to the root
    if my_rank==0:
      remote_final = comm.recv(source=num_ranks-1,tag=build_seq_tag)
      return remote_final
    elif my_rank==num_ranks-1:
      final = self.getFinal()
      comm.send(final,dest=0,tag=build_seq_tag)

    return None

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
  return sz[0] 

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
