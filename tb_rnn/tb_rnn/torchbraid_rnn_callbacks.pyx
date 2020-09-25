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
import math
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

    u =  <object> vec_u
    temp = pyApp.eval(u,tstart,tstop,level)

    # u.tensor().copy_(temp.tensor())
    u_h,u_c = u.tensors()
    temp_h,temp_c = temp.tensors()
    u_h.copy_(temp_h)
    u_c.copy_(temp_c)

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

  pyApp = <object>app

  tensors_X = (<object> x).tensors()
  tensors_Y = (<object> y).tensors()
  ten_X_h, ten_X_c = tensors_X
  ten_Y_h, ten_Y_c = tensors_Y

  cdef np.ndarray[float,ndim=1] np_X_h
  cdef np.ndarray[float,ndim=1] np_X_c
  cdef np.ndarray[float,ndim=1] np_Y_h
  cdef np.ndarray[float,ndim=1] np_Y_c

  # cdef np.ndarray[float,ndim=1] np_X
  # cdef np.ndarray[float,ndim=1] np_Y
  cdef int sz

  with pyApp.timer("my_sum"):
    # Cast x and y as a PyBraid_Vector
    np_X_h = ten_X_h.numpy().ravel()
    np_X_c = ten_X_c.numpy().ravel()
    np_Y_h = ten_Y_h.numpy().ravel()
    np_Y_c = ten_Y_c.numpy().ravel()
    # np_X = (<object> x).tensor().numpy().ravel()
    # np_Y = (<object> y).tensor().numpy().ravel()
    sz = len(np_X_h)
    # in place copy 
    for k in range(sz):
      np_Y_h[k] = alpha*np_X_h[k]+beta*np_Y_h[k]
      np_Y_c[k] = alpha*np_X_c[k]+beta*np_Y_c[k]
      # np_Y[k] = alpha*np_X[k]+beta*np_Y[k]

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
    # ten_U = (<object> u).tensor()
    # norm_ptr[0] = torch.norm(ten_U)
    tensors_U = (<object> u).tensors()
    norm_ptr[0] = 0.0
    for ten_U in tensors_U:
      norm_ptr[0] += torch.norm(ten_U)**2

    math.sqrt(norm_ptr[0])

  return 0

cdef int my_bufsize(braid_App app, int *size_ptr, braid_BufferStatus status):
  pyApp = <object> app
  cdef int cnt 
  with pyApp.timer("my_bufsize"):
    cnt = 0
    tensors_g0 = (<object> pyApp.g0).tensors()
    for ten_g0 in tensors_g0:
      cnt += ten_g0.size().numel()

    print("final cnt: ",cnt)

    # cnt = pyApp.x0.tensor().size().numel()

    # Note size_ptr is an integer array of size 1, and we index in at location [0]
    # the int size encodes the level
    size_ptr[0] = sizeof(float)*cnt + sizeof(float) + sizeof(int)
                   # vector                 time             level

  return 0

cdef int my_bufpack(braid_App app, braid_Vector u, void *buffer,braid_BufferStatus status):

  # Convert void * to a double array (note fbuffer is a C-array, so no bounds checking is done) 
  cdef int * ibuffer = <int *> buffer
  cdef float * fbuffer = <float *>(buffer+sizeof(int))
  cdef np.ndarray[float,ndim=1] np_U_h
  cdef np.ndarray[float,ndim=1] np_U_c
  # cdef np.ndarray[float,ndim=1] np_U
  cdef int sz_h
  cdef int sz_c
  cdef view.array my_buf 

  pyApp = <object>app
  with pyApp.timer("my_bufpack"):
    # Cast u as a PyBraid_Vector
    tensors_U = (<object> u).tensors()
    ten_U_h, ten_U_c = tensors_U
    # ten_U = (<object> u).tensor()
    np_U_h  = ten_U_h.numpy().ravel()
    np_U_c  = ten_U_c.numpy().ravel()
    # np_U  = ten_U.numpy().ravel() # ravel provides a flatten accessor to the array

    ibuffer[0] = (<object> u).level()
    fbuffer[0] = (<object> u).getTime()

    sz_h = len(np_U_h)
    sz_c = len(np_U_c)
    # sz = len(np_U)

    # TODO: Need to replace fbuffer with my_buf?
    # fbuffer size correct?
    # for k in range(sz_h):
    #   fbuffer[k+1] = np_U_h[k]

    # for k in range(sz_c):
    #   fbuffer[sz_h+k+1] = np_U_c[k]

    my_buf = <float[:sz_h]> (fbuffer+1)
    my_buf[:] = np_U_h

    my_buf = <float[:sz_c]> (fbuffer+1+sz_h)
    my_buf[:] = np_U_c

    # my_buf = <float[:sz]> (fbuffer+1)
    # my_buf[:] = np_U

  return 0

cdef int my_bufunpack(braid_App app, void *buffer, braid_Vector *u_ptr,braid_BufferStatus status):
  pyApp = <object>app

  cdef int * ibuffer = <int *> buffer
  cdef float * fbuffer = <float *>(buffer+sizeof(int))

  cdef np.ndarray[float,ndim=1] np_U_h
  cdef np.ndarray[float,ndim=1] np_U_c
  # cdef np.ndarray[float,ndim=1] np_U
  cdef int sz_h
  cdef int sz_c

  cdef view.array my_buf

  # cdef braid_Vector c_x = <braid_Vector> pyApp.g0
  # my_clone(app,c_x,u_ptr)

  # with pyApp.timer("my_bufunpack"):
  
  # allocate memory
  u_obj = pyApp.g0.clone()

  # u_obj = pyApp.x0.clone()
  Py_INCREF(u_obj) # why do we need this?
  u_ptr[0] = <braid_Vector> u_obj 

  u_obj.level_ = ibuffer[0]
  u_obj.setTime(fbuffer[0])

  tensors_U = u_obj.tensors()
  ten_U_h, ten_U_c = tensors_U

  np_U_h  = ten_U_h.numpy().ravel()
  np_U_c  = ten_U_c.numpy().ravel()

  # ten_U = u_obj.tensor()
  # np_U = ten_U.numpy().ravel() # ravel provides a flatten accessor to the array

  # this is almost certainly slow
  sz_h = len(np_U_h)
  sz_c = len(np_U_c)

  # print("sz_h: ", sz_h)
  # print("sz_c: ", sz_c)

  # TODO: Need to replace fbuffer with my_buf?
  # fbuffer size correct?
  # for k in range(sz_h):
  #   np_U_h[k] = fbuffer[k+1]

  # for k in range(sz_c):
  #   np_U_c[k] = fbuffer[sz_h+k+1]

  # my_buf = <float[:sz_h]> (fbuffer+1)
  # np_U_h = my_buf[:]

  # print("my_bufunpack - 8")

  # my_buf = <float[:sz_c]> (fbuffer+1+sz_h)
  # np_U_c = my_buf[:]

  # cdef int sz = len(np_U_h)
  for k in range(sz_h):
    np_U_h[k] = fbuffer[k+1]

  # cdef int sz_c = len(np_U_c)
  for k in range(sz_c):
    np_U_c[k] = fbuffer[sz_h+k+1]

  # print("ten_U_h: ",ten_U_h)
  # print("ten_U_c: ",ten_U_c)

    # sz = len(np_U)
    # my_buf = <float[:sz]> (fbuffer+1)
    # np_U[:] = my_buf

  return 0
