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

from cpython.mem cimport PyMem_Malloc, PyMem_Free
from cpython.ref cimport PyObject

import torch

include "../torchbraid_app.pyx"

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
  return <object> v_vec

# This builds a close of the initial vector
# using the `my_clone` 
def cloneVector(app,x):

  cdef braid_App c_app = <PyObject*>app
  cdef braid_Vector c_x = <braid_Vector> x
  cdef braid_Vector v 
  my_clone(c_app,c_x,&v)

  return <object> v

def addVector(app,alpha,ten_x,beta,ten_y):
  x = BraidVector(ten_x)
  y = BraidVector(ten_y)

  cdef braid_App c_app = <PyObject*>app
  cdef double dalpha = alpha
  cdef braid_Vector c_x = <braid_Vector>x
  cdef double dbeta  = beta
  cdef braid_Vector c_y = <braid_Vector>y

  my_sum(c_app,dalpha,c_x,dbeta,c_y)

def vectorNorm(app,ten_x):
  x = BraidVector(ten_x)

  cdef braid_App c_app = <PyObject*>app
  cdef braid_Vector c_x = <braid_Vector>x
  cdef double [1] norm = [ 0.0 ]
  
  my_norm(c_app,c_x,norm)

  return norm[0]

def bufSize(app):
  cdef braid_App c_app = <PyObject*>app
  cdef int [1] sz = [0]
  cdef _braid_BufferStatus_struct status
  
  my_bufsize(c_app,sz,&status)

  return sz[0] 

def sizeof_float(dtype):
  return int(torch.finfo(dtype).bits/8)

def eps_float(dtype):
  return torch.finfo(dtype).eps

cdef class MemoryBlock:
  cdef void* data
  cdef bint use_cuda
  cdef object app

  def __cinit__(self, app, size_t number):
    cdef uintptr_t addr

    self.app = app
    self.use_cuda = app.use_cuda

    float_type = app.dtype

    if app.use_cuda:
      addr = app.addBufferEntry(tensor=torch.empty(int(number/sizeof_float(float_type)), dtype=float_type, device='cuda'))
      self.data = <void *> addr
    else:
      # allocate some memory (uninitialised, may contain arbitrary data)
      self.data = <double*> PyMem_Malloc(number * sizeof(float_type))

    if not self.data:
      raise MemoryError()

  def __dealloc__(self):
    cdef uintptr_t addr

    if not self.use_cuda:
      PyMem_Free(self.data)  # no-op if self.data is NULL
    else:
      addr = <uintptr_t> self.data
      self.app.removeBufferEntry(addr=addr)

def pack(app,vec,block,level):
  cdef braid_App c_app    = <PyObject*>app
  cdef braid_Vector c_vec = <braid_Vector> vec
  cdef _braid_BufferStatus_struct status
  cdef MemoryBlock blk = <MemoryBlock> block

  my_bufpack(c_app, c_vec, blk.data,&status)

def unpack(app,block):
  cdef braid_App c_app    = <PyObject*>app
  cdef braid_Vector c_vec    
  cdef MemoryBlock blk = <MemoryBlock> block
  cdef _braid_BufferStatus_struct status
  
  my_bufunpack(c_app,blk.data,&c_vec,&status)

  vec = <object> c_vec
  return vec
