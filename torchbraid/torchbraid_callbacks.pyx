# cython: profile=True
# cython: linetrace=True

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

  # Create Numpy wrapper around u.v
  ten_u = <object> u

  braid_AccessStatusGetT(status, &t)

  pyApp.access(t,ten_u)
  return 0

cdef int my_step(braid_App app, braid_Vector ustop, braid_Vector fstop, braid_Vector vec_u, braid_StepStatus status):
  pyApp = <object> app
  u =  <object> vec_u

  cdef double tstart
  cdef double tstop
  cdef int level
  tstart = 0.0
  tstop = 5.0
  level = -1
  braid_StepStatusGetTstartTstop(status, &tstart, &tstop)
  braid_StepStatusGetLevel(status, &level)

  temp = pyApp.eval(u,tstart,tstop,level)
  u.tensor().copy_(temp.tensor())

  return 0
# end my_access

cdef int my_init(braid_App app, double t, braid_Vector *u_ptr):
  pyApp = <object> app
  u_mem = pyApp.buildInit(t)
  Py_INCREF(u_mem) # why do we need this?
  u_ptr[0] = <braid_Vector> u_mem
  return 0

cdef int my_free(braid_App app, braid_Vector u):
  # Cast u as a PyBraid_Vector
  pyU = <object> u
  # Decrement the smart pointer
  Py_DECREF(pyU) 
  del pyU
  return 0

cdef int my_sum(braid_App app, double alpha, braid_Vector x, double beta, braid_Vector y):
  py_app = <object>app

  cdef np.ndarray[float,ndim=1] np_X
  cdef np.ndarray[float,ndim=1] np_Y
  cdef int sz

  with py_app.timer("my_sum"):
    # Cast x and y as a PyBraid_Vector
    #np_X = (<object> x).tensor().numpy().ravel()
    #np_Y = (<object> y).tensor().numpy().ravel()
    #sz = len(np_X)

    ## in place copy (this is inefficient because of the copy/allocation to ten_T
    #for k in range(sz):
    #  np_Y[k] = alpha*np_X[k]+beta*np_Y[k]

    tn_X = (<object> x).tensor()
    tn_Y = (<object> y).tensor()

    tn_Y.mul_(beta)
    tn_Y.add_(tn_X,alpha)

  return 0

cdef int my_clone(braid_App app, braid_Vector u, braid_Vector *v_ptr):
  ten_U = <object> u 
  v_mem = ten_U.clone()
  Py_INCREF(v_mem) # why do we need this?
  v_ptr[0] = <braid_Vector> v_mem

  return 0

cdef int my_norm(braid_App app, braid_Vector u, double *norm_ptr):
  # Compute norm 
  ten_U = (<object> u).tensor()
  norm_ptr[0] = torch.norm(ten_U)

  return 0

cdef int my_bufsize(braid_App app, int *size_ptr, braid_BufferStatus status):
  pyApp = <object> app
  cdef int cnt = pyApp.x0.tensor().size().numel()

  # Note size_ptr is an integer array of size 1, and we index in at location [0]
  # the int size encodes the level
  size_ptr[0] = sizeof(float)*cnt + sizeof(float) + sizeof(int)
              # vector                 time             level

  return 0

cdef int my_bufpack(braid_App app, braid_Vector u, void *buffer,braid_BufferStatus status):

  # Convert void * to a double array (note fbuffer is a C-array, so no bounds checking is done) 
  cdef int * ibuffer = <int *> buffer
  cdef float * fbuffer = <float *>(buffer+sizeof(int))
  cdef np.ndarray[float,ndim=1] np_U
  cdef int sz
  cdef view.array my_buf 

  py_app = <object>app
  with py_app.timer("my_bufpack"):
    # Cast u as a PyBraid_Vector
    ten_U = (<object> u).tensor()
    np_U  = ten_U.numpy().ravel() # ravel provides a flatten accessor to the array

    ibuffer[0] = (<object> u).level()
    fbuffer[0] = (<object> u).getTime()

    sz = len(np_U)
    my_buf = <float[:sz]> (fbuffer+1)

    my_buf[:] = np_U

  return 0

cdef int my_bufunpack(braid_App app, void *buffer, braid_Vector *u_ptr,braid_BufferStatus status):
  py_app = <object>app

  cdef int * ibuffer = <int *> buffer
  cdef float * fbuffer = <float *>(buffer+sizeof(int))
  cdef braid_Vector c_x = <braid_Vector> py_app.x0
  cdef np.ndarray[float,ndim=1] np_U
  cdef int sz
  cdef view.array my_buf 

  with py_app.timer("my_bufunpack"):
  
    my_clone(app,c_x,u_ptr)
  
    (<object> u_ptr[0]).level_ = ibuffer[0]
    (<object> u_ptr[0]).setTime(fbuffer[0])

    ten_U = (<object> u_ptr[0]).tensor()
    np_U = ten_U.numpy().ravel() # ravel provides a flatten accessor to the array
    sz = len(np_U)
  
    # this is almost certainly slow
    my_buf = <float[:sz]> (fbuffer+1)
    np_U[:] = my_buf
  return 0
