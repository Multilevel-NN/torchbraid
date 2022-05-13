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
import pickle
cimport numpy as np

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
       
      # modify the state vector in place
      u =  <object> vec_u
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
      cl = BraidVector(tensors,ten_U.level_,ten_U.layer_data_,ten_U.send_flag_)
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
      norm_ptr[0] = 0.0
      for ten_U in tensors_U:
        val = torch.norm(ten_U).item()
        norm_ptr[0] += val*val

      norm_ptr[0] = math.sqrt(norm_ptr[0])
  except:
    output_exception("my_norm")

  return 0

cdef int my_bufsize(braid_App app, int *size_ptr, braid_BufferStatus status):

  try:
    pyApp = <object> app
    with pyApp.timer("bufsize"):
      shapes = pyApp.getTensorShapes()
      num_tensors = len(shapes) # all tensors
      cnt = 0
      total_shape = 0
      layer_data_size = pyApp.getLayerDataSize()
      for s in shapes:
        cnt += s.numel() # pyApp.shape0[0].numel()
        rank = len(s) #len(pyApp.shape0[0])
        total_shape += rank*sizeof(int)
  
      # because the braid vectors are sometimes moved with weight components, the app
      # object is responsible for making sure those are sized appropriately. 
  
      # Note size_ptr is an integer array of size 1, and we index in at location [0]
  
      # there are mulitple fields in a packed buffer, in order
      size_ptr[0] = ( sizeof(int)              # number of floats
                    + sizeof(int)              # level
                    + sizeof(int)              # num tensors
                    + sizeof(int)              # num weight tensors
                    + sizeof(int)              # other layer information (size in bytes)
                    + num_tensors*sizeof(int)  # tensor rank
                    + total_shape              # tensor shapes
                    + sizeof(float)*cnt        # tensor data
                    + layer_data_size          # pickled layer size
                    )
  except:
    output_exception("my_bufsize")

  return 0

cdef int my_bufpack(braid_App app, braid_Vector u, void *buffer,braid_BufferStatus status):

  # Convert void * to a double array (note fbuffer is a C-array, so no bounds checking is done) 
  cdef int * ibuffer
  cdef float * fbuffer
  cdef char * cbuffer 
  cdef np.ndarray[float,ndim=1] np_U
  cdef int offset
  cdef int foffset
  cdef int final_offset
  cdef int float_cnt 
  cdef view.array my_buf
  cdef float[:] fbuf_mv
  cdef float[:] np_U_mv 

  try:
    pyApp = <object> app
    with pyApp.timer("bufpack"):
      bv_u = <object> u

      all_tensors = bv_u.allTensors()
      flat_tensor = torch.cat([t.flatten() for t in all_tensors])
      float_cnt = flat_tensor.shape[0]

      # get the data copy started
      offset = 5 # this is accomdating space for the header integers
      fbuffer = <float *>(buffer+offset*sizeof(int)) 

      flat_tensor_cpu = torch.from_numpy(np.asarray(<float[:float_cnt]> fbuffer)) # allocated on host
      with pyApp.timer("bufpack-copy"): 
        flat_tensor_cpu.copy_(flat_tensor,non_blocking=True)                      # copied to host

      ibuffer = <int *> buffer
    
      # write out the buffer meta data
      level              = bv_u.level()
      num_tensors        = len(all_tensors)
      num_weight_tensors = len(bv_u.weightTensors())
      layer_data_size    = pyApp.getLayerDataSize()

      ibuffer[0] = level
      ibuffer[1] = num_tensors
      ibuffer[2] = num_weight_tensors
      ibuffer[3] = float_cnt
      ibuffer[4] = layer_data_size
    
      if bv_u.getLayerData() is not None: 
        pbuf_src = pickle.dumps(bv_u.getLayerData()) 
        assert(layer_data_size>=len(pbuf_src))
    
        cbuffer = <char *>(buffer+offset*sizeof(int)+float_cnt*sizeof(float)) 
    
        my_buf = <char[:len(pbuf_src)]> cbuffer
        my_buf[:] = pbuf_src

        final_offset = offset*sizeof(int)+float_cnt*sizeof(float)+layer_data_size*sizeof(char)
      else:
        final_offset = offset*sizeof(int)+float_cnt*sizeof(float)
      # end if layer_data_size

      ibuffer = <int *> (buffer+final_offset)
      offset = 0
      for t in all_tensors:
        size = t.size() 
        ibuffer[offset] = len(size)
        for i,s in enumerate(size):
          ibuffer[i+offset+1] = s
            
        offset += len(size)+1
      # end for a: creating space for the number tensors

  except:
    output_exception("my_bufpack")


  return 0

cdef int my_bufunpack(braid_App app, void *buffer, braid_Vector *u_ptr,braid_BufferStatus status):
  cdef int * ibuffer 
  cdef float * fbuffer 
  cdef void * vbuffer 
  cdef np.ndarray[float,ndim=1] np_U
  cdef int offset
  cdef int float_cnt
  cdef int sz
  cdef int layer_data_size
  cdef view.array my_buf 

  try:
    pyApp = <object>app
    with pyApp.timer("bufunpack"):

      # read in the buffer metda data
      ibuffer = <int *> buffer
    
      level              = ibuffer[0]
      num_tensors        = ibuffer[1]
      num_weight_tensors = ibuffer[2]
      float_cnt          = ibuffer[3]
      layer_data_size    = ibuffer[4]
    
      offset = 5
      # copy from the buffer into the braid vector
      fbuffer = <float *>(buffer+(offset)*sizeof(int)) # level, rank, sizes

      sizes = []
      ibuffer = <int *> (buffer+offset*sizeof(int)+float_cnt*sizeof(float)+layer_data_size*sizeof(char))
      offset = 0
      for t in range(num_tensors):
        rank = ibuffer[offset]
        size = rank*[0]
        for i in range(rank):
          size[i] = ibuffer[i+offset+1]
            
        sizes += [size]
        offset += len(size)+1

      # build up the braid vector
      tens = [] #torch.zeros(s) for s in sizes]
    
      for s in sizes:
        ten_U = torch.zeros(s)
        np_U = ten_U.numpy().ravel() # ravel provides a flatten accessor to the array
      
        # copy the buffer into the tensor
        sz = len(np_U)
        my_buf = <float[:sz]> (fbuffer)
        np_U[:] = my_buf
        
        # update the float buffer pointer
        fbuffer = <float*> (fbuffer+sz)
        if hasattr(pyApp,'device'):
          ten_U = ten_U.to(pyApp.device,non_blocking=False)
        tens += [ten_U]
      # end for s

      # build an vector object and set the tensors to land in the correct places
      vector_tensors = tens[0:num_tensors-num_weight_tensors]
      weight_tensors = tens[num_tensors-num_weight_tensors:]

      layer_data = None
      if layer_data_size>0:
        # this is to make sure I can use the vbuffer
        vbuffer = fbuffer
    
        my_buf = <char[:layer_data_size]> vbuffer
        layer_data = pickle.loads(my_buf)
      # end if layer_data_size

      u_obj = BraidVector(vector_tensors,level,layer_data,send_flag=True)
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
    cu_vec = BraidVector(cu_mem,level)
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
    fu_vec = BraidVector(fu_mem,level)
    Py_INCREF(fu_vec) # why do we need this?

    fu_ptr[0] = <braid_Vector> fu_vec

  return 0
