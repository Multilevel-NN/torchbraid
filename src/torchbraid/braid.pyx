from torchbraid.braid_funcs cimport *
cdef class PyBraid_Core:
    '''
    This class is a simple wrapper about the C-struct braid_Core
    This class allows us to return a braid_Core back into Python,
    and then have Python give that braid_Core back to Braid. 
    
    Note, that you cannot pass C-structs like braid_Core directly back to
    Python
    '''
    cdef braid_Core core

    def __init__(self): 
        pass

    cdef setCore(self, braid_Core _core):
        self.core = _core
    
    cdef braid_Core getCore(self):
        return self.core


# cdef double* PyBraid_VoidToDoubleArray(void *buffer):
#     '''
#     Convert void * array to a double array 
#     Note dbuffer is a C-array, so no bounds checking is done
#     '''
#     cdef double *dbuffer = <double*> buffer
#     return dbuffer 
# 
# 
# cdef int* PyBraid_VoidToIntArray(void *buffer):
#     '''
#     Convert void * array to a double array 
#     Note ibuffer is a C-array, so no bounds checking is done
#     '''
#     cdef int *ibuffer = <int*> buffer
#     return ibuffer 

