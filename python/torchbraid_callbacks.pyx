import torch
import numpy as np

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from cpython.ref cimport PyObject, Py_INCREF, Py_DECREF

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

cdef int my_step(braid_App app, braid_Vector ustop, braid_Vector fstop, braid_Vector u, braid_StepStatus status):
  pyApp = <object> app
  ten_u =  <object> u

  cdef double tstart
  cdef double tstop
  tstart = 0.0
  tstop = 5.0
  braid_StepStatusGetTstartTstop(status, &tstart, &tstop)

  temp = pyApp.eval(ten_u,tstart,tstop)
  ten_u[:] = temp[:]

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
  # Cast x and y as a PyBraid_Vector
  ten_X = <object> x
  ten_Y = <object> y
  ten_T = ten_Y.clone()

  # in place copy (this is inefficient because of the copy/allocation to ten_T
  ten_Y[:] = alpha*ten_X+beta*ten_T

  return 0

cdef int my_clone(braid_App app, braid_Vector u, braid_Vector *v_ptr):
  ten_U = <object> u 
  v_mem = ten_U.clone()
  Py_INCREF(v_mem) # why do we need this?
  v_ptr[0] = <braid_Vector> v_mem

  return 0

cdef int my_norm(braid_App app, braid_Vector u, double *norm_ptr):
  # Compute norm 
  ten_U = <object> u
  norm_ptr[0] = torch.norm(ten_U)

  return 0

cdef int my_bufsize(braid_App app, int *size_ptr, braid_BufferStatus status):
  pyApp = <object> app
  cdef int cnt = pyApp.x0.size().numel()

  #  Note size_ptr is an integer array of size 1, and we index in at location [0]
  size_ptr[0] = sizeof(double)*cnt

  return 0

cdef int my_bufpack(braid_App app, braid_Vector u, void *buffer,braid_BufferStatus status):
    # Cast u as a PyBraid_Vector
    ten_U = <object> u
    np_U  = ten_U.numpy() 

    # Convert void * to a double array (note dbuffer is a C-array, so no bounds checking is done) 
    cdef double * dbuffer = <double*>(buffer)
    cdef int dval = <int>(dbuffer)
    cdef int val = <int>(buffer)

    # Pack buffer
    k = 0
    for item in np_U.flat: 
      dbuffer[k] = item
      k += 1
    # end for item

    return 0

cdef int my_bufunpack(braid_App app, void *buffer, braid_Vector *u_ptr,braid_BufferStatus status):
  py_app = <object>app

  cdef double * dbuffer = <double*> buffer
  cdef braid_Vector c_x = <PyObject*>py_app.x0

  my_clone(app,c_x,u_ptr)

  ten_U = <object> u_ptr[0]
  np_U  = ten_U.numpy().ravel() # ravel provides a flatten accessor to the array

  # this is almost certainly slow
  for k in range(len(np_U)):
    np_U[k] = dbuffer[k]

  return 0
