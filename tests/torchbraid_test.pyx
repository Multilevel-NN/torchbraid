# cython: profile=True
# cython: linetrace=True

from cpython.mem cimport PyMem_Malloc, PyMem_Free
from cpython.ref cimport PyObject

include "../torchbraid/torchbraid_app.pyx"

##
# Define your Python Braid Vector

# to supress a warning from numpy
cdef extern from *:
  """
  #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
  """

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
  return (<object> v_vec).tensor()

# This builds a close of the initial vector
# using the `my_clone` 
def cloneVector(app,x):
  b_vec = BraidVector(x,0)

  cdef braid_App c_app = <PyObject*>app
  cdef braid_Vector c_x = <braid_Vector> b_vec
  cdef braid_Vector v 
  my_clone(c_app,c_x,&v)

  return (<object> v).tensor()

def addVector(app,alpha,ten_x,beta,ten_y):
  x = BraidVector(ten_x,0)
  y = BraidVector(ten_y,0)

  cdef braid_App c_app = <PyObject*>app
  cdef double dalpha = alpha
  cdef braid_Vector c_x = <braid_Vector>x
  cdef double dbeta  = beta
  cdef braid_Vector c_y = <braid_Vector>y

  my_sum(c_app,dalpha,c_x,dbeta,c_y)

def vectorNorm(app,ten_x):
  x = BraidVector(ten_x,0)

  cdef braid_App c_app = <PyObject*>app
  cdef braid_Vector c_x = <braid_Vector>x
  cdef double [1] norm = [ 0.0 ]
  
  my_norm(c_app,c_x,norm)

  return norm[0]

def bufSize(app):
  cdef braid_App c_app = <PyObject*>app
  cdef int [1] sz = [0]
  cdef braid_BufferStatus status = NULL
  
  my_bufsize(c_app,sz,status)

  # subtract the int size (for testing purposes)
  return sz[0] 

def allocBuffer(app):
  cdef void * buffer = PyMem_Malloc(bufSize(app))
  return <object> buffer

def freeBuffer(app,obuffer):
  cdef void * buffer = <void*> obuffer
  PyMem_Free(buffer)

def pack(app,ten_vec,obuffer,level):
  vec = BraidVector(ten_vec,level)

  cdef braid_App c_app    = <PyObject*>app
  cdef braid_Vector c_vec = <braid_Vector> vec
  cdef braid_BufferStatus status = NULL
  cdef void * buffer = <void*> obuffer

  my_bufpack(c_app, c_vec, buffer,status)

def unpack(app,obuffer):
  cdef braid_App c_app    = <PyObject*>app
  cdef braid_Vector c_vec    
  cdef void * buffer = <void*> obuffer
  cdef braid_BufferStatus status = NULL
  
  my_bufunpack(c_app,buffer,&c_vec,status)

  vec = <object> c_vec
  return (vec.tensor(),vec.level())
