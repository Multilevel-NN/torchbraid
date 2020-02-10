import torch
import numpy as np
cimport numpy as np

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

cdef int my_step(braid_App app, braid_Vector ustop, braid_Vector fstop, braid_Vector vec_u, braid_StepStatus status):
  pyApp = <object> app
  u =  <object> vec_u

  cdef double tstart
  cdef double tstop
  tstart = 0.0
  tstop = 5.0
  braid_StepStatusGetTstartTstop(status, &tstart, &tstop)

  temp = pyApp.eval(u,tstart,tstop)
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
  # Cast x and y as a PyBraid_Vector
  cdef np.ndarray[float,ndim=1] np_X = (<object> x).tensor().numpy().ravel()
  cdef np.ndarray[float,ndim=1] np_Y = (<object> y).tensor().numpy().ravel()

  # in place copy (this is inefficient because of the copy/allocation to ten_T
  cdef int sz = len(np_X)
  for k in range(sz):
    np_Y[k] = alpha*np_X[k]+beta*np_Y[k]

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
  size_ptr[0] = sizeof(double)*cnt + sizeof(int)

  return 0

cdef int my_bufpack(braid_App app, braid_Vector u, void *buffer,braid_BufferStatus status):
    # Cast u as a PyBraid_Vector
    ten_U = (<object> u).tensor()
    cdef np.ndarray[float,ndim=1] np_U  = ten_U.numpy().ravel() # ravel provides a flatten accessor to the array

    # Convert void * to a double array (note dbuffer is a C-array, so no bounds checking is done) 
    cdef int * ibuffer = <int *> buffer
    cdef double * dbuffer = <double *>(buffer+4)

    ibuffer[0] = (<object> u).level()

    # Pack buffer
    cdef int sz = len(np_U)
    for k in range(sz):
      dbuffer[k] = np_U[k]
    # end for item

    return 0

cdef int my_bufunpack(braid_App app, void *buffer, braid_Vector *u_ptr,braid_BufferStatus status):
  py_app = <object>app

  cdef int * ibuffer = <int *> buffer
  cdef double * dbuffer = <double *>(buffer+4)
  cdef braid_Vector c_x = <braid_Vector> py_app.x0

  my_clone(app,c_x,u_ptr)

  (<object> u_ptr[0]).level_ = ibuffer[0]
  ten_U = (<object> u_ptr[0]).tensor()
  cdef np.ndarray[float,ndim=1] np_U = ten_U.numpy().ravel() # ravel provides a flatten accessor to the array

  # this is almost certainly slow
  cdef int sz = len(np_U)
  for k in range(sz):
    np_U[k] = dbuffer[k]

  return 0

cdef int my_coarsen(braid_App app, braid_Vector vec_fu, braid_Vector *cu_ptr, braid_CoarsenRefStatus status):
  pyApp  = <object> app
  fu =  <object> vec_fu

  cdef int level = -1

  cu_mem = pyApp.coarsen(fu.tensor(),fu.level())

  cu = BraidVector(cu_mem,fu.level()+1)
  Py_INCREF(cu) # why do we need this?

  cu_ptr[0] = <braid_Vector> cu

  return 0

cdef int my_refine(braid_App app, braid_Vector cu_vec, braid_Vector *fu_ptr, braid_CoarsenRefStatus status):
  pyApp  = <object> app
  cu =  <object> cu_vec

  cdef int level = -1

  fu_mem = pyApp.refine(cu.tensor(),cu.level())
  fu = BraidVector(fu_mem,cu.level()-1)
  Py_INCREF(fu) # why do we need this?

  fu_ptr[0] = <braid_Vector> fu

  return 0
