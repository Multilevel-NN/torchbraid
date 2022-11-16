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

def output_exception(label):
  s = traceback.format_exc()
  print('\n**** Torchbraid Callbacks::{} Exception ****\n{}'.format(label,s))

##
# Define your Python Braid Vector as a C-struct

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
      if u.hasStream():
        with pyApp.timer("step-synch"):
          u.syncStream()

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
  except:
    output_exception("my_sum")
    sys.stdout.flush()
    sys.exit(1)

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

  try:
    pyApp = <object> app
    with pyApp.timer("bufsize"):
      if not pyApp.gpu_direct_commu or not pyApp.use_cuda:
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

        # there are mulitple fields in a packed buffer, in order
        size_ptr[0] = (sizeof(int)  # number of floats
                       + sizeof(int)  # num tensors
                       + sizeof(int)  # num weight tensors
                       + sizeof(int)  # other layer information (size in bytes)
                       + num_tensors * sizeof(int)  # tensor rank
                       + total_shape  # tensor shapes
                       + sizeof(float) * cnt  # tensor data
                       )
      else:
        #TODO: Is there a nicer way to get the buff_elements number?
        tmp = pyApp.getTensorShapes()[1:]
        buff_elements = np.sum([item.numel() for item in tmp])
        size_ptr[0] = (buff_elements * sizeof(float))


  except:
    output_exception("my_bufsize")

  return 0

cdef int my_bufpack(braid_App app, braid_Vector u, void *buffer,braid_BufferStatus status):

  # Convert void * to a double array (note fbuffer is a C-array, so no bounds checking is done)
  cdef int * ibuffer
  cdef float * fbuffer
  cdef char * cbuffer
  cdef int head_offset # header offset
  cdef int foffset
  cdef int final_offset
  cdef int float_cnt
  cdef view.array my_buf
  cdef uintptr_t addr


  try:
    pyApp = <object> app
    if not pyApp.gpu_direct_commu or not pyApp.use_cuda:
      with pyApp.timer("bufpack"):
        bv_u = <object> u

        all_tensors = bv_u.allTensors()
        with pyApp.timer("bufpack-flatten"):
          flat_tensor = torch.cat([t.detach().flatten() for t in all_tensors])
        float_cnt = flat_tensor.shape[0]

        # get the data copy started
        with pyApp.timer("bufpack-copy"):
          if pyApp.use_cuda:
            flat_tensor_cpu = flat_tensor.to(torch.device('cpu'),non_blocking=True) # copied to host
          else:
            flat_tensor_cpu = flat_tensor

        # write out the buffer header data
        head_offset = 3 # this is accomdating space for the header integers
        with pyApp.timer("bufpack-header"):
          ibuffer = <int *> buffer

          num_tensors        = len(all_tensors)
          num_weight_tensors = len(bv_u.weightTensors())

          ibuffer[0] = num_tensors
          ibuffer[1] = num_weight_tensors
          ibuffer[2] = float_cnt

        final_offset = head_offset*sizeof(int)+float_cnt*sizeof(float)

        with pyApp.timer("bufpack-sizes"):
          ibuffer = <int *> (buffer+final_offset)
          offset = 0
          for t in all_tensors:
            size = t.size()
            ibuffer[offset] = len(size)
            for i,s in enumerate(size):
              ibuffer[i+offset+1] = s

            offset += len(size)+1
          # end for a: creating space for the number tensors

        with pyApp.timer("bufpack-synch"):
          if pyApp.use_cuda:
            torch.cuda.synchronize()

        with pyApp.timer("bufpack-tonumpy"):
          fbuffer = <float *>(buffer+head_offset*sizeof(int))

          flat_tensor_numpy = torch.from_numpy(np.asarray(<float[:float_cnt]> fbuffer))
          flat_tensor_numpy.copy_(flat_tensor_cpu)
    else:
      with pyApp.timer("bufpack"):
        addr = <uintptr_t>buffer
        app_buffer = pyApp.getBuffer(addr = addr)

        bv_u = <object> u

        #with pyApp.timer("bufpack-synch"):
        #  if pyApp.use_cuda:
        #    torch.cuda.synchronize()

        all_tensors = bv_u.allTensors()
        start = 0
        for item in all_tensors:
          flat = item.detach().flatten()
          size = flat.shape[0]
          app_buffer[start:start + size] = flat
          start += size

  except:
    output_exception("my_bufpack")


  return 0

cdef int my_bufunpack(braid_App app, void *buffer, braid_Vector *u_ptr,braid_BufferStatus status):
  pyApp = <object> app
  if pyApp.use_cuda:
    return my_bufunpack_cuda(app, buffer, u_ptr,status)
  else:
    return my_bufunpack_cpu(app, buffer, u_ptr,status)
# end my_bufunpack

cdef int my_bufunpack_cuda(braid_App app, void *buffer, braid_Vector *u_ptr,braid_BufferStatus status):
  cdef int * ibuffer
  cdef float * fbuffer
  cdef void * vbuffer
  cdef int head_offset
  cdef int float_cnt
  cdef int sz
  cdef view.array my_buf
  cdef uintptr_t addr

  try:
    pyApp = <object> app
    if not pyApp.gpu_direct_commu or not pyApp.use_cuda:
      with pyApp.timer("bufunpack"):

        # read in the buffer metda data
        ibuffer = <int *> buffer

        num_tensors        = ibuffer[0]
        num_weight_tensors = ibuffer[1]
        float_cnt          = ibuffer[2]

        head_offset = 3

        if not hasattr(pyApp,'stream'):
          pyApp.stream = Stream(pyApp.device)
        stream = pyApp.stream

        # copy from the buffer into the braid vector
        with torch.cuda.stream(stream):
          fbuffer = <float *>(buffer+(head_offset)*sizeof(int)) # rank, sizes
          with pyApp.timer("bufunpack-move"):
            ten_cpu = torch.from_numpy(np.asarray(<float[:float_cnt]> fbuffer))
            ten_gpu = ten_cpu.to(pyApp.device)

          with pyApp.timer("bufunpack-sizes"):
            sizes = []
            ibuffer = <int *> (buffer+head_offset*sizeof(int)+float_cnt*sizeof(float))
            offset = 0
            for t in range(num_tensors):
              rank = ibuffer[offset]
              size = rank*[0]
              for i in range(rank):
                size[i] = ibuffer[i+offset+1]

              sizes += [torch.Size(size)]
              offset += len(size)+1

          with pyApp.timer("bufunpack-movearound"):
            tens = []
            i0 = 0
            for s in sizes:
              i1 = i0+s.numel()
              tens += [ten_gpu[i0:i1].reshape(s)]
              i0 = i1

          # build an vector object and set the tensors to land in the correct places
          with pyApp.timer("bufunpack-wrap"):
            vector_tensors = tens[0:num_tensors-num_weight_tensors]
            weight_tensors = tens[num_tensors-num_weight_tensors:]

            u_obj = BraidVector(vector_tensors,send_flag=True)
            u_obj.weight_tensor_data_ = weight_tensors
            Py_INCREF(u_obj)

            # set the pointer for output
            u_ptr[0] = <braid_Vector> u_obj

        u_obj.setStream(stream)
    else:
      with pyApp.timer("bufunpack"):
        addr = <uintptr_t>buffer
        app_buffer = pyApp.getBuffer(addr)

        vt = []
        wt = []
        shapes = pyApp.getTensorShapes()[1:]
        ten_sizes = [item.numel() for item in shapes]
        __fake_t__ = 0.0 

        size_wt = len(pyApp.getParameterShapes(t=__fake_t__))
        size_vt = len(pyApp.shape0[1:])

        size = 0
        for i in range(len(shapes)):
          if i < size_vt:
            vt.append(torch.reshape(app_buffer[size:size+ten_sizes[i]], shapes[i]))
          elif i < size_vt+size_wt:
            wt.append(torch.reshape(app_buffer[size:size+ten_sizes[i]], shapes[i]))
          else:
            raise Exception(f'Bufunpack: size vt {size_vt} + size wt {size_wt} != len(shapes) {len(shapes)}')
          size += ten_sizes[i]

        if not hasattr(pyApp,'stream'):
          pyApp.stream = Stream(pyApp.device)
        stream = pyApp.stream

        #TODO: Get the level information, get the layer data
        u_obj = BraidVector(tensor = vt, send_flag = True)
        u_obj.weight_tensor_data_ = wt
        Py_INCREF(u_obj)

        # set the pointer for output
        u_ptr[0] = <braid_Vector> u_obj

        u_obj.setStream(stream)

  except:
    output_exception("my_bufunpack_gpu")

  return 0

cdef int my_bufunpack_cpu(braid_App app, void *buffer, braid_Vector *u_ptr,braid_BufferStatus status):
  cdef int * ibuffer
  cdef float * fbuffer
  cdef void * vbuffer
  cdef int head_offset
  cdef int float_cnt
  cdef int sz
  cdef view.array my_buf

  try:
    pyApp = <object>app
    with pyApp.timer("bufunpack"):

      # read in the buffer metda data
      ibuffer = <int *> buffer

      num_tensors        = ibuffer[0]
      num_weight_tensors = ibuffer[1]
      float_cnt          = ibuffer[2]

      head_offset = 3

      # copy from the buffer into the braid vector
      fbuffer = <float *>(buffer+(head_offset)*sizeof(int)) # rank, sizes
      with pyApp.timer("bufunpack-move"):
        ten_cpu = torch.from_numpy(np.asarray(<float[:float_cnt]> fbuffer))
        ten_cpu = ten_cpu.detach().clone()

      with pyApp.timer("bufunpack-sizes"):
        sizes = []
        ibuffer = <int *> (buffer+head_offset*sizeof(int)+float_cnt*sizeof(float))
        offset = 0
        for t in range(num_tensors):
          rank = ibuffer[offset]
          size = rank*[0]
          for i in range(rank):
            size[i] = ibuffer[i+offset+1]

          sizes += [torch.Size(size)]
          offset += len(size)+1

      with pyApp.timer("bufunpack-movearound"):
        tens = []
        i0 = 0
        for s in sizes:
          i1 = i0+s.numel()
          tens += [ten_cpu[i0:i1].reshape(s)]
          i0 = i1

      # build an vector object and set the tensors to land in the correct places
      with pyApp.timer("bufunpack-wrap"):
        vector_tensors = tens[0:num_tensors-num_weight_tensors]
        weight_tensors = tens[num_tensors-num_weight_tensors:]

        u_obj = BraidVector(vector_tensors,send_flag=True)
        u_obj.weight_tensor_data_ = weight_tensors
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

cdef int my_bufalloc(braid_App app, void **buffer, int nbytes):
  cdef uintptr_t addr

  pyApp = <object>app

  if not pyApp.gpu_direct_commu or not pyApp.use_cuda:
    buffer[0] = malloc(nbytes)
  else:
    tmp = pyApp.getTensorShapes()[1:]
    buff_elements = np.sum([item.numel() for item in tmp])
    addr = pyApp.addBufferEntry(tensor=torch.empty(buff_elements, dtype=torch.float32, device='cuda'))

    buffer[0]=<void *> addr
  return 0

cdef int my_buffree(braid_App app, void **buffer):
  cdef uintptr_t addr

  pyApp = <object> app
  if not pyApp.gpu_direct_commu or not pyApp.use_cuda:
    free(buffer[0])
    buffer[0] = NULL
  else:
    addr = <uintptr_t> buffer[0]
    pyApp.removeBufferEntry(addr=addr)
  return 0
