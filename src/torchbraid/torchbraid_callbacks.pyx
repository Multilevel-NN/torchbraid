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
import traceback
import sys
cimport numpy as np

from torch.cuda import Stream

from libc.stdint cimport uintptr_t
from libc.stdlib cimport malloc, free
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from cpython.ref cimport PyObject, Py_INCREF, Py_DECREF
from cython cimport view

__float_alloc_type__ = torch.float32

def output_exception(label):
  s = traceback.format_exc()
  print('\n**** Torchbraid Callbacks::{} Exception ****\n{}'.format(label,s))

##
# Define your Python Braid Vector as a C-struct

cdef int get_bytes(dtype):
  return int(torch.finfo(dtype).bits/8)

cdef int my_access(braid_App app,braid_Vector u,braid_AccessStatus status):

  cdef double t

  try:
    pyApp = <object> app
    with pyApp.timer("access"):

      # Create Numpy wrapper around u.v
      ten_u = <object> u

      braid_AccessStatusGetT(status, &t)

      pyApp.access(t,ten_u)
  except:
    output_exception("my_access")

  return 0

cdef int my_step(braid_App app, braid_Vector ustop, braid_Vector fstop, braid_Vector vec_u, braid_StepStatus status):
  cdef double tstart
  cdef double tstop
  cdef int tindex
  cdef int level
  cdef int done
  #cdef int sidx

  try:
    pyApp = <object> app

    with pyApp.timer("step"):

      tstart = 0.0
      tstop = 5.0
      level = -1
      braid_StepStatusGetTstartTstop(status, &tstart, &tstop)
      braid_StepStatusGetLevel(status, &level)
      braid_StepStatusGetDone(status, &done)

      u = <object> vec_u

      # modify the state vector in place
      pyApp.eval(u,tstart,tstop,level,done)

      # store final step
      if level==0 and tstop==pyApp.Tf:
        pyApp.x_final = u.clone()

  except:
    output_exception("my_step: rank={}, step=({},{}), level={}, sf={}".format(pyApp.getMPIComm().Get_rank(),
                                                                                           tstart,
                                                                                           tstop,
                                                                                           level,
                                                                                           u.getSendFlag()))

  return 0
# end my_access

cdef int my_init(braid_App app, double t, braid_Vector *u_ptr):

  try:
    pyApp = <object> app
    with pyApp.timer("init"):
      u_mem = pyApp.buildInit(t)
      Py_INCREF(u_mem) # why do we need this?

      u_ptr[0] = <braid_Vector> u_mem

      # finish the step computation
      if pyApp.use_cuda:
        torch.cuda.synchronize()
  except:
    output_exception("my_init")

  return 0

cdef int my_free(braid_App app, braid_Vector u):
  try:
    pyApp = <object> app
    with pyApp.timer("free"):
      # Cast u as a PyBraid_Vector
      pyU = <object> u
      # Decrement the smart pointer
      Py_DECREF(pyU)
      del pyU
  except:
    output_exception("my_free")
  return 0

cdef int my_sum(braid_App app, double alpha, braid_Vector x, double beta, braid_Vector y):
  try:
    pyApp = <object> app
    with pyApp.timer("sum"):
      bv_X = <object> x
      bv_Y = <object> y

      for ten_X,ten_Y in zip(bv_X.tensors(),bv_Y.tensors()):
        ten_Y.mul_(float(beta))
        ten_Y.add_(ten_X,alpha=float(alpha))

      ## finish the sum computation
      ##if pyApp.use_cuda:
      ##  torch.cuda.synchronize()
  except:
    x_shapes = [ten_X.size() for ten_X in bv_X.tensors()]
    y_shapes = [ten_Y.size() for ten_Y in bv_Y.tensors()]

    output_exception(f"my_sum: {x_shapes}, {y_shapes}")
    sys.stdout.flush()

  return 0

cdef int my_clone(braid_App app, braid_Vector u, braid_Vector *v_ptr):
  try:
    pyApp = <object> app
    with pyApp.timer("clone"):
      ten_U = <object> u
      #v_mem = ten_U.clone()

      tensors = [t.detach().clone() for t in ten_U.tensor_data_]
      cl = BraidVector(tensors,ten_U.send_flag_)
      if len(ten_U.weight_tensor_data_)>0:
        cl.weight_tensor_data_ = [t.detach() for t in ten_U.weight_tensor_data_]

      v_mem = cl
      Py_INCREF(v_mem) # why do we need this?
      v_ptr[0] = <braid_Vector> v_mem
  except:
    output_exception("my_clone")

  return 0

cdef int my_norm(braid_App app, braid_Vector u, double *norm_ptr):
  try:
    pyApp = <object> app
    with pyApp.timer("norm"):
      # Compute norm
      tensors_U = (<object> u).tensors()
      norms = torch.stack([torch.square(ten_U).sum() for ten_U in tensors_U])

      norm_ptr[0] = math.sqrt(norms.sum().item())
  except:
    output_exception("my_norm")

  return 0

cdef int my_bufsize(braid_App app, int *size_ptr, braid_BufferStatus status):
  cdef int tidx
  cdef int level

  braid_BufferStatusGetTIndex(status, &tidx)
  braid_BufferStatusGetLevel(status, &level)

  try:
    pyApp = <object> app

    float_type = float # on CPU
    if pyApp.user_mpi_buf:
      float_type = __float_alloc_type__ # on CUDA
    with pyApp.timer("bufsize"):
      shapes = pyApp.getFeatureShapes(tidx,level) + pyApp.getParameterShapes(tidx,level)

      cnt = 0
      for s in shapes:
        cnt += s.numel() 
      size_ptr[0] = get_bytes(float_type)*cnt
  except:
    output_exception("my_bufsize")

  return 0

cdef int my_bufpack(braid_App app, braid_Vector u, void * buffer,braid_BufferStatus status):
  cdef int tidx
  cdef int level

  braid_BufferStatusGetTIndex(status, &tidx)
  braid_BufferStatusGetLevel(status, &level)

  pyApp = <object> app


  try:
    if pyApp.use_cuda:
      return my_bufpack_cuda(app, u, buffer, tidx, level)
    else:
      return my_bufpack_cpu(app, u, buffer, tidx, level)
  except:
    output_exception("my_bufpack")
# end my_bufpack

cdef int my_bufpack_cpu(braid_App app, braid_Vector u, void *buffer,int tidx, int level):
  cdef int start

  try:
    pyApp = <object> app
    with pyApp.timer("bufpack"):
      bv_u = <object> u

      all_tensors = bv_u.allTensors()

      start = 0
      for item in all_tensors:
        flat = item.detach().flatten()
        size = int(flat.shape[0])
        tbuffer = torch.from_numpy(np.asarray(<float[:size]> (buffer+start*get_bytes(float))))
        tbuffer.copy_(flat)
        start += size

  except:
    output_exception("my_bufpack_cpu")

  return 0

cdef int my_bufpack_cuda(braid_App app, braid_Vector u, void *buffer,int tidx, int level):

  # Convert void * to a double array (note fbuffer is a C-array, so no bounds checking is done)
  cdef uintptr_t addr

  try:
    pyApp = <object> app
    with pyApp.timer("bufpack"):

      addr = <uintptr_t> buffer
      app_buffer = pyApp.getBuffer(addr = addr)

      bv_u = <object> u

      all_tensors = bv_u.allTensors()
      start = 0
      for item in all_tensors:
        flat = item.detach().flatten()
        size = flat.shape[0]
        app_buffer[start:start + size] = flat
        start += size

      # finish the data movement
      torch.cuda.synchronize()

  except:
    output_exception(f"my_bufpack_cuda: time index = {tidx}, level = {level}")

  return 0

cdef int my_bufunpack(braid_App app, void *buffer, braid_Vector *u_ptr,braid_BufferStatus status):
  cdef int tidx
  cdef int level

  braid_BufferStatusGetTIndex(status, &tidx)
  braid_BufferStatusGetLevel(status, &level)

  pyApp = <object> app

  try:
    if pyApp.use_cuda:
      result = my_bufunpack_cuda(app, buffer, u_ptr,tidx, level)
    else:
      result = my_bufunpack_cpu(app, buffer, u_ptr,tidx, level)
  except:
    output_exception("my_bufunpack")

  return result
# end my_bufunpack

cdef int my_bufunpack_cuda(braid_App app, void *buffer, braid_Vector *u_ptr,int tidx,int level):
  cdef uintptr_t addr

  try:
    pyApp = <object> app
    with pyApp.timer("bufunpack"):
      addr = <uintptr_t> buffer
      app_buffer = pyApp.getBuffer(addr = addr)

      size_vt = pyApp.getFeatureShapes(tidx,level)
      size_wt = pyApp.getParameterShapes(tidx,level)

      vt = []
      start = 0
      for s in size_vt:
        size = s.numel()
        vt.append(torch.reshape(app_buffer[start:start+size].detach().clone(), s))
        start += size

      wt = []
      for s in size_wt:
        size = s.numel()
        wt.append(torch.reshape(app_buffer[start:start+size].detach().clone(), s))
        start += size

      u_obj = BraidVector(tensor = vt, send_flag = True)
      u_obj.weight_tensor_data_ = wt
      Py_INCREF(u_obj)

      # set the pointer for output
      u_ptr[0] = <braid_Vector> u_obj

      # finish data movement (this one might not be neccessary)
      torch.cuda.synchronize()
  except:
    output_exception("my_bufunpack_gpu")

  return 0

cdef int my_bufunpack_cpu(braid_App app, void *buffer, braid_Vector *u_ptr,int tidx,int level):
  cdef int start

  try:
    pyApp = <object>app

    with pyApp.timer("bufunpack"):
      size_vt = pyApp.getFeatureShapes(tidx,level)
      size_wt = pyApp.getParameterShapes(tidx,level)

      vt = []
      start = 0
      for s in size_vt:
        size = s.numel()
        tbuffer = torch.from_numpy(np.asarray(<float[:size]> (buffer+start*get_bytes(float))))
        vt.append(torch.reshape(tbuffer.detach().clone(), s))
        start += size

      wt = []
      for s in size_wt:
        size = s.numel()
        tbuffer = torch.from_numpy(np.asarray(<float[:size]> (buffer+start*get_bytes(float))))
        wt.append(torch.reshape(tbuffer.detach().clone(), s))
        start += size

      u_obj = BraidVector(tensor = vt, send_flag = True)
      u_obj.weight_tensor_data_ = wt
      Py_INCREF(u_obj)

      # set the pointer for output
      u_ptr[0] = <braid_Vector> u_obj
  except:
    output_exception("my_bufunpack")

  return 0

cdef int my_coarsen(braid_App app, braid_Vector fu, braid_Vector *cu_ptr, braid_CoarsenRefStatus status):
  cdef int level = -1

  pyApp  = <object> app
  with pyApp.timer("coarsen"):
    ten_fu =  (<object> fu).tensor()

    braid_CoarsenRefStatusGetLevel(status,&level)

    cu_mem = pyApp.spatial_coarse(ten_fu,level)
    cu_vec = BraidVector(cu_mem)
    Py_INCREF(cu_vec) # why do we need this?

    cu_ptr[0] = <braid_Vector> cu_vec

  return 0

cdef int my_refine(braid_App app, braid_Vector cu, braid_Vector *fu_ptr, braid_CoarsenRefStatus status):
  cdef int level = -1

  pyApp  = <object> app
  with pyApp.timer("refine"):
    ten_cu =  (<object> cu).tensor()

    braid_CoarsenRefStatusGetNRefine(status,&level)

    fu_mem = pyApp.spatial_refine(ten_cu,level)
    fu_vec = BraidVector(fu_mem)
    Py_INCREF(fu_vec) # why do we need this?

    fu_ptr[0] = <braid_Vector> fu_vec

  return 0

cdef int my_bufalloc(braid_App app, void **buffer, int nbytes, braid_BufferStatus status):
  cdef uintptr_t addr

  pyApp = <object>app

  if pyApp.use_cuda:
    # convert nbytes to number of elements
    elements = math.ceil(nbytes / get_bytes(__float_alloc_type__))

    addr = pyApp.addBufferEntry(tensor=torch.empty(elements, dtype=__float_alloc_type__, device='cuda'))

    buffer[0]=<void *> addr

    torch.cuda.synchronize()
  else:
    buffer[0] = malloc(nbytes)

  return 0

cdef int my_buffree(braid_App app, void **buffer):
  cdef uintptr_t addr

  pyApp = <object> app
  if pyApp.use_cuda:
    addr = <uintptr_t> buffer[0]
    pyApp.removeBufferEntry(addr=addr)

  else:
    free(buffer[0])
    buffer[0] = NULL

  return 0
