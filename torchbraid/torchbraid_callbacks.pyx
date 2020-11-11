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

import math
import torch
import numpy as np
cimport numpy as np

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from cpython.ref cimport PyObject, Py_INCREF, Py_DECREF
from cython cimport view

##
# Define your Python Braid Vector as a C-struct

cdef int my_access(braid_App app,braid_Vector u,braid_AccessStatus status):
  pyApp = <object> app

  cdef double t

  with pyApp.timer("my_access"):
    # Create Numpy wrapper around u.v
    ten_u = <object> u

    braid_AccessStatusGetT(status, &t)

    pyApp.access(t,ten_u)
  return 0

cdef int my_step(braid_App app, braid_Vector ustop, braid_Vector fstop, braid_Vector vec_u, braid_StepStatus status):
  pyApp = <object> app

  cdef double tstart
  cdef double tstop
  cdef int level

  with pyApp.timer("my_step"):

    tstart = 0.0
    tstop = 5.0
    level = -1
    braid_StepStatusGetTstartTstop(status, &tstart, &tstop)
    braid_StepStatusGetLevel(status, &level)

    # modify the state vector in place
    u =  <object> vec_u
    pyApp.eval(u,tstart,tstop,level)

  return 0
# end my_access

cdef int my_init(braid_App app, double t, braid_Vector *u_ptr):
  pyApp = <object> app
  with pyApp.timer("my_init"):
    u_mem = pyApp.buildInit(t)
    Py_INCREF(u_mem) # why do we need this?
    u_ptr[0] = <braid_Vector> u_mem

  return 0

cdef int my_free(braid_App app, braid_Vector u):
  pyApp = <object> app
  with pyApp.timer("my_free"):
    # Cast u as a PyBraid_Vector
    pyU = <object> u
    # Decrement the smart pointer
    Py_DECREF(pyU) 
    del pyU
  return 0

cdef int my_sum(braid_App app, double alpha, braid_Vector x, double beta, braid_Vector y):
  # This routine cna be made faster by using the pyTorch tensor operations
  # My initial attempt at this failed however

  with torch.no_grad():
    bv_X = <object> x
    bv_Y = <object> y
    for ten_X,ten_Y in zip(bv_X.tensors(),bv_Y.tensors()):
      ten_Y.mul_(float(beta))
      ten_Y.add_(ten_X,alpha=float(alpha))

  return 0

cdef int my_clone(braid_App app, braid_Vector u, braid_Vector *v_ptr):
  pyApp = <object> app
  with pyApp.timer("my_clone"):
    ten_U = <object> u 
    v_mem = ten_U.clone()
    Py_INCREF(v_mem) # why do we need this?
    v_ptr[0] = <braid_Vector> v_mem

  return 0

cdef int my_norm(braid_App app, braid_Vector u, double *norm_ptr):
  pyApp = <object> app
  with pyApp.timer("my_norm"):
    # Compute norm 
    tensors_U = (<object> u).tensors()
    norm_ptr[0] = 0.0
    for ten_U in tensors_U:
      norm_ptr[0] += torch.dot(ten_U,ten_U)

    math.sqrt(norm_ptr[0])

  return 0

cdef int my_bufsize(braid_App app, int *size_ptr, braid_BufferStatus status):
  pyApp = <object> app

  shapes = pyApp.getTensorShapes()
  num_tensors = len(shapes) # all tensors
  cnt = 0
  total_shape = 0
  for s in shapes:
    cnt += s.numel() # pyApp.shape0[0].numel()
    rank = len(s) #len(pyApp.shape0[0])
    total_shape += rank*sizeof(int)

  # because the braid vectors are sometimes moved with weight components, the app
  # object is responsible for making sure those are sized appropriately. 

  # Note size_ptr is an integer array of size 1, and we index in at location [0]

  # there are mulitple fields in a packed buffer, in orderr
  #     level (1 int), num_tensors (1 int), num_weight_tensors (1 int), [rank (1 int), sizes (rank int)] * (num_tensors), tensor data

  size_ptr[0] = sizeof(int) + sizeof(int) + sizeof(int) + num_tensors*sizeof(int) + total_shape + sizeof(float)*cnt
                 #  level     num_tensors   num_weight_tensors          rank             total_shape         vector_data

  return 0

cdef int my_bufpack(braid_App app, braid_Vector u, void *buffer,braid_BufferStatus status):

  # Convert void * to a double array (note fbuffer is a C-array, so no bounds checking is done) 
  cdef int * ibuffer
  cdef float * fbuffer
  cdef np.ndarray[float,ndim=1] np_U
  cdef int offset
  cdef int sz
  cdef view.array my_buf 

  bv_u = <object> u

  ibuffer = <int *> buffer

  # write out the buffer meta data
  level              = bv_u.level()
  num_tensors        = len(bv_u.allTensors())
  num_weight_tensors = len(bv_u.weightTensors())

  ibuffer[0] = level
  ibuffer[1] = num_tensors
  ibuffer[2] = num_weight_tensors

  offset = 3 # this is accomdating space for the three integers
  for t in bv_u.allTensors():
    size = t.size() 
    ibuffer[offset] = len(size)
    for i,s in enumerate(size):
      ibuffer[i+offset+1] = s
        
    offset += len(size)+1
  # end for a: creating space for the number tensors

  # copy the data
  fbuffer = <float *>(buffer+offset*sizeof(int)) 
  for ten_U in bv_u.allTensors():
    np_U  = ten_U.numpy().ravel() # ravel provides a flatten accessor to the array

    # copy the tensor into the buffer
    sz = len(np_U)
    my_buf = <float[:sz]> (fbuffer)
    my_buf[:] = np_U

    # update the float buffer pointer
    fbuffer = <float*> (fbuffer+sz)

  return 0

cdef int my_bufunpack(braid_App app, void *buffer, braid_Vector *u_ptr,braid_BufferStatus status):
  pyApp = <object>app

  cdef int * ibuffer 
  cdef float * fbuffer 
  cdef np.ndarray[float,ndim=1] np_U
  cdef int offset
  cdef int sz
  cdef view.array my_buf 

  # read in the buffer metda data
  ibuffer = <int *> buffer

  level              = ibuffer[0]
  num_tensors        = ibuffer[1]
  num_weight_tensors = ibuffer[2]

  offset = 3
  sizes = []
  for t in range(num_tensors):
    rank = ibuffer[offset]
    size = rank*[0]
    for i in range(rank):
      size[i] = ibuffer[i+offset+1]
        
    sizes += [size]
    offset += len(size)+1

  # build up the braid vector
  tens = [torch.zeros(s) for s in sizes]
  vector_tensors = tens[0:num_tensors-num_weight_tensors]
  weight_tensors = tens[num_tensors-num_weight_tensors:-1]

  # build an vector object and set the tensors to land in the correct places
  u_obj = BraidVector(tuple(vector_tensors),level)
  Py_INCREF(u_obj) 
  u_obj.addWeightTensors(weight_tensors)

  # copy from the buffer into the braid vector
  fbuffer = <float *>(buffer+(offset)*sizeof(int)) # level, rank, sizes
  for ten_U in u_obj.allTensors():
    np_U = ten_U.numpy().ravel() # ravel provides a flatten accessor to the array
  
    # copy the buffer into the tensor
    sz = len(np_U)
    my_buf = <float[:sz]> (fbuffer)
    np_U[:] = my_buf
    
    # update the float buffer pointer
    fbuffer = <float*> (fbuffer+sz)

  # set the pointer for output
  u_ptr[0] = <braid_Vector> u_obj 

  return 0

cdef int my_coarsen(braid_App app, braid_Vector fu, braid_Vector *cu_ptr, braid_CoarsenRefStatus status):
  pyApp  = <object> app
  ten_fu =  (<object> fu).tensor()

  cdef int level = -1
  braid_CoarsenRefStatusGetLevel(status,&level)

  cu_mem = pyApp.spatial_coarse(ten_fu,level)
  cu_vec = BraidVector(cu_mem,level)
  Py_INCREF(cu_vec) # why do we need this?

  cu_ptr[0] = <braid_Vector> cu_vec

  return 0

cdef int my_refine(braid_App app, braid_Vector cu, braid_Vector *fu_ptr, braid_CoarsenRefStatus status):
  pyApp  = <object> app
  ten_cu =  (<object> cu).tensor()

  cdef int level = -1
  braid_CoarsenRefStatusGetNRefine(status,&level)

  fu_mem = pyApp.spatial_refine(ten_cu,level)
  fu_vec = BraidVector(fu_mem,level)
  Py_INCREF(fu_vec) # why do we need this?

  fu_ptr[0] = <braid_Vector> fu_vec

  return 0
