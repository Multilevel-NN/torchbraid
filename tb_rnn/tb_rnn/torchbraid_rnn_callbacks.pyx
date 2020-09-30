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

cdef int rnn_my_step(braid_App app, braid_Vector ustop, braid_Vector fstop, braid_Vector vec_u, braid_StepStatus status):
  pyApp = <object> app

  cdef double tstart
  cdef double tstop
  cdef int level

  with pyApp.timer("rnn_my_step"):

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

cdef int rnn_my_bufsize(braid_App app, int *size_ptr, braid_BufferStatus status):
  pyApp = <object> app
  cdef int cnt 
  with pyApp.timer("rnn_my_bufsize"):
    cnt = 0
    tensors_g0 = (<object> pyApp.g0).tensors()
    for ten_g0 in tensors_g0:
      cnt += ten_g0.size().numel()

    print("final cnt: ",cnt)

    # cnt = pyApp.x0.tensor().size().numel()

    # Note size_ptr is an integer array of size 1, and we index in at location [0]
    # the int size encodes the level
    size_ptr[0] = sizeof(float)*cnt + sizeof(int)
                   # vector                 level

  return 0

cdef int rnn_my_bufpack(braid_App app, braid_Vector u, void *buffer,braid_BufferStatus status):

  # Convert void * to a double array (note fbuffer is a C-array, so no bounds checking is done) 
  cdef int * ibuffer = <int *> buffer
  cdef float * fbuffer = <float *>(buffer+sizeof(int))
  cdef np.ndarray[float,ndim=1] np_U_h
  cdef np.ndarray[float,ndim=1] np_U_c
  # cdef np.ndarray[float,ndim=1] np_U
  cdef int sz_h
  cdef int sz_c
  cdef view.array rnn_my_buf 

  pyApp = <object>app
  with pyApp.timer("rnn_my_bufpack"):
    # Cast u as a PyBraid_Vector
    tensors_U = (<object> u).tensors()
    ten_U_h, ten_U_c = tensors_U
    # ten_U = (<object> u).tensor()
    np_U_h  = ten_U_h.numpy().ravel()
    np_U_c  = ten_U_c.numpy().ravel()
    # np_U  = ten_U.numpy().ravel() # ravel provides a flatten accessor to the array

    ibuffer[0] = (<object> u).level()

    sz_h = len(np_U_h)
    sz_c = len(np_U_c)

    rnn_my_buf = <float[:sz_h]> (fbuffer)
    rnn_my_buf[:] = np_U_h

    rnn_my_buf = <float[:sz_c]> (fbuffer+sz_h)
    rnn_my_buf[:] = np_U_c

  return 0

cdef int rnn_my_bufunpack(braid_App app, void *buffer, braid_Vector *u_ptr,braid_BufferStatus status):
  pyApp = <object>app

  cdef int * ibuffer = <int *> buffer
  cdef float * fbuffer = <float *>(buffer+sizeof(int))

  cdef np.ndarray[float,ndim=1] np_U_h
  cdef np.ndarray[float,ndim=1] np_U_c
  # cdef np.ndarray[float,ndim=1] np_U
  cdef int sz_h
  cdef int sz_c

  cdef view.array rnn_my_buf

  # allocate memory
  u_obj = pyApp.g0.clone()

  Py_INCREF(u_obj) # why do we need this?
  u_ptr[0] = <braid_Vector> u_obj 

  u_obj.level_ = ibuffer[0]

  tensors_U = u_obj.tensors()
  ten_U_h, ten_U_c = tensors_U

  np_U_h  = ten_U_h.numpy().ravel()
  np_U_c  = ten_U_c.numpy().ravel()

  # ten_U = u_obj.tensor()
  # np_U = ten_U.numpy().ravel() # ravel provides a flatten accessor to the array

  # this is almost certainly slow
  sz_h = len(np_U_h)
  sz_c = len(np_U_c)

  # cdef int sz = len(np_U_h)
  for k in range(sz_h):
    np_U_h[k] = fbuffer[k]

  # cdef int sz_c = len(np_U_c)
  for k in range(sz_c):
    np_U_c[k] = fbuffer[sz_h+k]

  return 0
