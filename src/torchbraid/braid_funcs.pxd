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

'''
Cython header file defining the Braid-Python interface
'''

cimport mpi4py.libmpi as libmpi
from cpython.ref cimport PyObject
from libc.stdio cimport FILE

ctypedef PyObject _braid_App_struct
ctypedef _braid_App_struct * braid_App

ctypedef PyObject _braid_Vector_struct
ctypedef _braid_Vector_struct *braid_Vector

cdef extern from "status.h":
    ##
    # Import the user-interface status structure definitions
    cdef struct _braid_Status_struct:
        pass
    ctypedef _braid_Status_struct *braid_Status

    cdef struct _braid_AccessStatus_struct:
        pass
    ctypedef _braid_AccessStatus_struct *braid_AccessStatus

    cdef struct _braid_SyncStatus_struct:
        pass
    ctypedef _braid_SyncStatus_struct *braid_SyncStatus

    cdef struct _braid_StepStatus_struct:
        pass
    ctypedef _braid_StepStatus_struct *braid_StepStatus

    cdef struct _braid_BufferStatus_struct:
        pass
    ctypedef _braid_BufferStatus_struct *braid_BufferStatus

    cdef struct _braid_CoarsenRefStatus_struct:
        pass
    ctypedef _braid_CoarsenRefStatus_struct *braid_CoarsenRefStatus


cdef extern from "braid_status.h":
    ##
    # Wrap StepStatus Routines
    int braid_StepStatusGetT(braid_StepStatus status, double *t_ptr)
    int braid_StepStatusGetTIndex(braid_StepStatus status, int *idx_ptr)
    int braid_StepStatusGetIter(braid_StepStatus status, int *iter_ptr)
    int braid_StepStatusGetLevel(braid_StepStatus status, int *level_ptr)
    int braid_StepStatusGetNLevels(braid_StepStatus status, int *nlevels_ptr)
    int braid_StepStatusGetNRefine(braid_StepStatus status, int *nrefine_ptr)
    int braid_StepStatusGetNTPoints(braid_StepStatus status, int *ntpoints_ptr)
    int braid_StepStatusGetTstop(braid_StepStatus status, double *tstop_ptr)
    int braid_StepStatusGetTstartTstop(braid_StepStatus status, double *tstart_ptr, double *tstop_ptr)
    int braid_StepStatusGetTol(braid_StepStatus status, double *tol_ptr)
    int braid_StepStatusGetRNorms(braid_StepStatus status, int *nrequest_ptr, double *rnorms_ptr)
    int braid_StepStatusGetOldFineTolx(braid_StepStatus status, double *old_fine_tolx_ptr)
    int braid_StepStatusSetOldFineTolx(braid_StepStatus status, double old_fine_tolx)
    int braid_StepStatusSetTightFineTolx(braid_StepStatus status, double tight_fine_tolx)
    int braid_StepStatusSetRFactor(braid_StepStatus status, double rfactor)
    int braid_StepStatusSetRSpace(braid_StepStatus status, double r_space)
    int braid_StepStatusGetDone(braid_StepStatus status, int *done)

    ##
    # Wrap AccessStatus Routines
    int braid_AccessStatusGetT(braid_AccessStatus status, double *t_ptr)
    int braid_AccessStatusGetTIndex(braid_AccessStatus status, int *idx_ptr)
    int braid_AccessStatusGetIter(braid_AccessStatus status, int *iter_ptr)
    int braid_AccessStatusGetLevel(braid_AccessStatus status, int *level_ptr)
    int braid_AccessStatusGetNLevels(braid_AccessStatus status, int *nlevels_ptr)
    int braid_AccessStatusGetNRefine(braid_AccessStatus status, int *nrefine_ptr)
    int braid_AccessStatusGetNTPoints(braid_AccessStatus status, int *ntpoints_ptr)
    int braid_AccessStatusGetResidual(braid_AccessStatus status, double *rnorm_ptr)
    int braid_AccessStatusGetDone(braid_AccessStatus status, int *done_ptr)
    int braid_AccessStatusGetTILD(braid_AccessStatus status, double *t_ptr, int *iter_ptr, int *level_ptr,
                                  int *done_ptr)
    int braid_AccessStatusGetWrapperTest(braid_AccessStatus status, int *wtest_ptr)
    int braid_AccessStatusGetCallingFunction(braid_AccessStatus status, int *cfunction_ptr)

    ##
    # Wrap CoarsenRefStatus Routines
    int braid_CoarsenRefStatusGetT(braid_CoarsenRefStatus status, double *t_ptr)
    int braid_CoarsenRefStatusGetTIndex(braid_CoarsenRefStatus status, int *idx_ptr)
    int braid_CoarsenRefStatusGetIter(braid_CoarsenRefStatus status, int *iter_ptr)
    int braid_CoarsenRefStatusGetLevel(braid_CoarsenRefStatus status, int *level_ptr)
    int braid_CoarsenRefStatusGetNLevels(braid_CoarsenRefStatus status, int *nlevels_ptr)
    int braid_CoarsenRefStatusGetNRefine(braid_CoarsenRefStatus status, int *nrefine_ptr)
    int braid_CoarsenRefStatusGetNTPoints(braid_CoarsenRefStatus status, int *ntpoints_ptr)
    int braid_CoarsenRefStatusGetCTprior(braid_CoarsenRefStatus status, double *ctprior_ptr)
    int braid_CoarsenRefStatusGetCTstop(braid_CoarsenRefStatus status, double *ctstop_ptr)
    int braid_CoarsenRefStatusGetFTprior(braid_CoarsenRefStatus status, double *ftprior_ptr)
    int braid_CoarsenRefStatusGetFTstop(braid_CoarsenRefStatus status, double *ftstop_ptr)
    int braid_CoarsenRefStatusGetTpriorTstop(braid_CoarsenRefStatus status, double *t_ptr, double *ftprior_ptr,
                                             double *ftstop_ptr, double *ctprior_ptr, double *ctstop_ptr)

    ##
    # Wrap BufferStatus Routines
    int braid_BufferStatusGetMessageType(braid_BufferStatus status, int *messagetype_ptr)
    int braid_BufferStatusSetSize(braid_BufferStatus status, int size)
    int braid_BufferStatusGetTIndex(braid_BufferStatus status, int * idx)
    int braid_BufferStatusGetLevel(braid_BufferStatus status, int * level)

    ##
    # Wrap SyncStatus Routines
    int braid_SyncStatusGetTIUL(braid_SyncStatus status, int *iloc_upper, int *iloc_lower, int level)
    int braid_SyncStatusGetTimeValues(braid_SyncStatus status, double ** tvalues_ptr, int i_upper, int i_lower,
                                      int level)
    int braid_SyncStatusGetIter(braid_SyncStatus status, int *iter_ptr)
    int braid_SyncStatusGetLevel(braid_CoarsenRefStatus status, int *level_ptr)
    int braid_SyncStatusGetNLevels(braid_CoarsenRefStatus status, int *nlevels_ptr)
    int braid_SyncStatusGetNRefine(braid_CoarsenRefStatus status, int *nrefine_ptr)
    int braid_SyncStatusGetNTPoints(braid_CoarsenRefStatus status, int *ntpoints_ptr)
    int braid_SyncStatusGetDone(braid_AccessStatus status, int *done_ptr)
    int braid_SyncStatusGetCallingFunction(braid_AccessStatus status, int *cfunction_ptr)

cdef extern from "braid.h":
    ##
    # Wrap Braid Core
    cdef struct _braid_Core_struct:
        int warm_restart
        _braid_Grid          ** grids

    ctypedef _braid_Core_struct *braid_Core

    ##
    # Wrap all the function pointers that User's can define
    ctypedef int (*braid_PtFcnStep)(braid_App app, braid_Vector ustop, braid_Vector fstop, braid_Vector u,
                                    braid_StepStatus status)

    ctypedef int (*braid_PtFcnInit)(braid_App app, double t, braid_Vector *u_ptr)

    ctypedef int (*braid_PtFcnClone)(braid_App app, braid_Vector u, braid_Vector *v_ptr)

    ctypedef int (*braid_PtFcnFree)(braid_App app, braid_Vector u)

    ctypedef int (*braid_PtFcnSum)(braid_App app, double alpha, braid_Vector x, double beta, braid_Vector y)

    ctypedef int (*braid_PtFcnSpatialNorm)(braid_App app, braid_Vector u, double *norm_ptr)

    ctypedef int (*braid_PtFcnAccess)(braid_App app, braid_Vector u, braid_AccessStatus status)

    ctypedef int (*braid_PtFcnBufSize)(braid_App app, int *size_ptr, braid_BufferStatus status)

    ctypedef int (*braid_PtFcnBufPack)(braid_App app, braid_Vector u, void *buffer, braid_BufferStatus status)

    ctypedef int (*braid_PtFcnBufUnpack)(braid_App app, void *buffer, braid_Vector *u_ptr, braid_BufferStatus status)

    ctypedef int (*braid_PtFcnBufAlloc)(braid_App app, void ** buffer, int nbytes)

    ctypedef int (*braid_PtFcnBufFree)(braid_App app, void ** buffer)

    ctypedef int (*braid_PtFcnResidual)(braid_App app, braid_Vector ustop, braid_Vector r, braid_StepStatus status)

    ctypedef int (*braid_PtFcnSCoarsen)(braid_App app, braid_Vector fu, braid_Vector *cu_ptr,
                                        braid_CoarsenRefStatus  status)

    ctypedef int (*braid_PtFcnSRefine)(braid_App app, braid_Vector cu, braid_Vector *fu_ptr,
                                       braid_CoarsenRefStatus  status)

    ctypedef int (*braid_PtFcnSInit)(braid_App app, double t, braid_Vector *u_ptr)

    ctypedef int (*braid_PtFcnSClone)(braid_App app, braid_Vector u, braid_Vector *v_ptr)

    ctypedef int (*braid_PtFcnSFree)(braid_App app, braid_Vector  u)

    ctypedef int (*braid_PtFcnTimeGrid)(braid_App app, double *ta, int *ilower, int *iupper)

    ##
    # Wrap BraidSet Routines
    int braid_PrintStats(braid_Core core)
    int braid_SetMaxLevels(braid_Core core, int max_levels)
    int braid_SetSkip(braid_Core core, int skip)
    int braid_SetRefine(braid_Core core, int refine)
    int braid_SetMaxRefinements(braid_Core core, int max_refinements)
    int braid_SetTPointsCutoff(braid_Core core, int tpoints_cutoff)
    int braid_SetMinCoarse(braid_Core core, int min_coarse)
    int braid_SetRelaxOnlyCG(braid_Core core, int relax_only_cg)
    int braid_SetAbsTol(braid_Core core, double atol)
    int braid_SetRelTol(braid_Core core, double rtol)
    int braid_SetNRelax(braid_Core core, int level, int nrelax)
    int braid_SetCRelaxWt(braid_Core  core, int level, double Cwt)
    int braid_SetCFactor(braid_Core core, int level, int cfactor)
    int braid_SetMaxIter(braid_Core core, int max_iter)
    int braid_SetFMG(braid_Core core)
    int braid_SetNFMG(braid_Core core, int k)
    int braid_SetNFMGVcyc(braid_Core core, int nfmg_Vcyc)
    int braid_SetStorage(braid_Core core, int storage)
    int braid_SetTemporalNorm(braid_Core core, int tnorm)
    int braid_SetResidual(braid_Core core, braid_PtFcnResidual residual)
    int braid_SetFullRNormRes(braid_Core core, braid_PtFcnResidual residual)
    int braid_SetTimeGrid(braid_Core core, braid_PtFcnTimeGrid tgrid)
    int braid_SetSpatialCoarsen(braid_Core core, braid_PtFcnSCoarsen scoarsen)
    int braid_SetSpatialRefine(braid_Core core, braid_PtFcnSRefine srefine)
    int braid_SetBufAllocFree(braid_Core core, braid_PtFcnBufAlloc bufalloc, braid_PtFcnBufFree buffree)
    int braid_SetPrintLevel(braid_Core core, int print_level)
    int braid_SetFileIOLevel(braid_Core core, int io_level)
    int braid_SetPrintFile(braid_Core core, const char *printfile_name)
    int braid_SetDefaultPrintFile(braid_Core core)
    int braid_SetAccessLevel(braid_Core core, int access_level)
    int braid_SplitCommworld(const libmpi.MPI_Comm *comm_world, int px, libmpi.MPI_Comm *comm_x,
                             libmpi.MPI_Comm *comm_t)
    int braid_SetShell(braid_Core core, braid_PtFcnSInit sinit, braid_PtFcnSClone sclone, braid_PtFcnSFree sfree)
    int braid_SetTimerFile(braid_Core core, int length, const char *filestem)
    int braid_ResetTimer(braid_Core core)
    int braid_SetTimings(braid_Core core, int timing_level)

    ##
    # Braid Test Routines
    int braid_TestInitAccess(braid_App app, libmpi.MPI_Comm comm_x, FILE *fp, double t,
                             braid_PtFcnInit init, braid_PtFcnAccess access, braid_PtFcnFree free)
    int braid_TestClone(braid_App app, libmpi.MPI_Comm comm_x, FILE *fp, double t,
                        braid_PtFcnInit init, braid_PtFcnAccess access,
                        braid_PtFcnFree free, braid_PtFcnClone clone)
    int braid_TestSum(braid_App app, libmpi.MPI_Comm comm_x, FILE *fp, double t,
                      braid_PtFcnInit  init, braid_PtFcnAccess access, braid_PtFcnFree free,
                      braid_PtFcnClone clone, braid_PtFcnSum sum)
    int braid_TestSpatialNorm(braid_App app, libmpi.MPI_Comm comm_x, FILE *fp, double t,
                              braid_PtFcnInit init, braid_PtFcnFree free, braid_PtFcnClone clone,
                              braid_PtFcnSum sum, braid_PtFcnSpatialNorm spatialnorm)
    int braid_TestBuf(braid_App app, libmpi.MPI_Comm comm_x, FILE *fp, double t,
                      braid_PtFcnInit init, braid_PtFcnFree free, braid_PtFcnSum sum,
                      braid_PtFcnSpatialNorm spatialnorm, braid_PtFcnBufSize bufsize,
                      braid_PtFcnBufPack bufpack, braid_PtFcnBufUnpack bufunpack)
    int braid_TestCoarsenRefine(braid_App app, libmpi.MPI_Comm comm_x, FILE *fp, double t, double fdt, double cdt,
                                braid_PtFcnInit init, braid_PtFcnAccess access, braid_PtFcnFree free,
                                braid_PtFcnClone clone, braid_PtFcnSum sum, braid_PtFcnSpatialNorm spatialnorm,
                                braid_PtFcnSCoarsen coarsen, braid_PtFcnSRefine refine)
    int braid_TestResidual(braid_App app, libmpi.MPI_Comm comm_x, FILE *fp, double t, double dt,
                           braid_PtFcnInit myinit, braid_PtFcnAccess myaccess, braid_PtFcnFree myfree,
                           braid_PtFcnClone clone, braid_PtFcnSum sum, braid_PtFcnSpatialNorm spatialnorm,
                           braid_PtFcnResidual residual, braid_PtFcnStep step)
    int braid_TestAll(braid_App app, libmpi.MPI_Comm comm_x, FILE *fp, double t, double fdt, double cdt,
                      braid_PtFcnInit init, braid_PtFcnFree free, braid_PtFcnClone clone, braid_PtFcnSum sum,
                      braid_PtFcnSpatialNorm spatialnorm, braid_PtFcnBufSize bufsize, braid_PtFcnBufPack bufpack,
                      braid_PtFcnBufUnpack bufunpack, braid_PtFcnSCoarsen coarsen,
                      braid_PtFcnSRefine refine, braid_PtFcnResidual residual, braid_PtFcnStep step)

    ##
    # Wrap BraidGet Routines
    int braid_GetNumIter(braid_Core core, int *iter_ptr)
    int braid_GetRNorms(braid_Core core, int *nrequest_ptr, double *rnorms)
    int braid_GetNLevels(braid_Core core, int *nlevels_ptr)
    int braid_GetSpatialAccuracy(braid_StepStatus status, double loose_tol, double tight_tol, double *tol_ptr)
    int braid_SetSeqSoln(braid_Core core, int seq_soln)
    int braid_GetMyID(braid_Core core, int *myid_ptr)

    int braid_SetRevertedRanks(braid_Core core, int boolean)

    int braid_SetFinalFCRelax(braid_Core core)

    ##
    # Wrap Braid Init, Drive, and Destroy
    int braid_Init(libmpi.MPI_Comm comm_world, libmpi.MPI_Comm comm,
                   double tstart, double tstop, int ntime, braid_App app,
                   braid_PtFcnStep step, braid_PtFcnInit init, braid_PtFcnClone clone,
                   braid_PtFcnFree free, braid_PtFcnSum sum, braid_PtFcnSpatialNorm spatialnorm,
                   braid_PtFcnAccess access, braid_PtFcnBufSize bufsize, braid_PtFcnBufPack bufpack,
                   braid_PtFcnBufUnpack bufunpack, braid_Core *core_ptr)

    ##
    # Run Braid
    int braid_Drive(braid_Core core)

    ##
    # Deallocation
    int braid_Destroy(braid_Core core)


cdef extern from "_braid.h":
    # Wrap Braid Core
    cdef struct _braid_BaseVector_struct:
        braid_Vector    userVector

    ctypedef _braid_BaseVector_struct *braid_BaseVector

    cdef struct _braid_Grid:
        int          level  # Level that grid is on
        int          ilower  # smallest time index at this level
        int          iupper  # largest time index at this level
        int          clower  # smallest C point index
        int          cupper  # largest C point index
        int          gupper  # global size of the grid
        int          cfactor  # coarsening factor
        int          ncpoints  # number of C points
        int          nupoints  # number of unknown vector points
        braid_BaseVector *ua  # unknown vectors  (C-points at least)
        double *ta  # time values                (all points)

        braid_BaseVector   ulast  # stores vector at last time step, only set in FAccess and FCRelax if done is True

    ##
    # helper function for initializing coarse vectors
    int _braid_InitGuess(braid_Core  core,
                         int   level)

    ##
    # helper function for accessing primal vectors, returns a reference
    int _braid_UGetVectorRef(braid_Core  core,
                             int         level,
                             int         index,
                             braid_BaseVector *u_ptr)

    ##
    # helper function for accessing primal vectors, returns a copy
    int _braid_UGetVector(braid_Core  core,
                          int         level,
                          int         index,
                          braid_BaseVector *u_ptr)
    ##
    # helper function to get F and C index information for "interval_index"
    # each processor has ncpoints total intervals
    int _braid_GetInterval(braid_Core   core,
                           int          level,
                           int          interval_index,
                           int *flo_ptr,
                           int *fhi_ptr,
                           int *ci_ptr)

    ##
    # helper function for getting the lower and upper time step indices
    int _braid_GetDistribution(braid_Core   core,
                               int *ilower_ptr,
                               int *iupper_ptr)

    ##
    # helper function to get the processor number that time step index resides on
    int _braid_GetProc(braid_Core   core,
                       int          level,
                       int          index,
                       int *proc_ptr)

cdef extern from "util.h":
    ## Switch for displaying the XBraid actions. Used for debugging only.
    int _braid_SetVerbosity(braid_Core  core,
                            int   verbose_adj)

# cdef object convert_carray_to_numpy(double * v, dim1, dim2=1, dim3=1):
#     '''
#     Helper function to cast C array v to an (dim1 x dim2 x dim3)
#     numpy array. Note, no copying of data is done.
#
#     Input
#     -----
#     v     <double *>:   C data array to cast
#     dim1  <int>     :   First dimension of v
#     dim2  <int>     :   Second dimension of v (optional)
#     dim3  <int>     :   Third dimension of v (optional)
#
#     Output
#     ----
#     v_view          :   Numpy array that points to same memory as input v
#
#     '''
#
#     assert(dim1 > 0), "Array dimensions must be positive, dim1=%d"%dim1
#     assert(dim2 >= 0), "Array dimensions must be positive, dim2=%d"%dim2
#     assert(dim3 >= 0), "Array dimensions must be positive, dim3=%d"%dim3
#
#     # Convert v to a linear array of size dim1*dim2*dim3
#     dim = dim1*dim2*dim3
#     cdef double[:] v_view = <double[:dim]> v
#     v_np = np.asarray(v_view)
#
#     if dim2 > 1 and dim3 == 1:
#        v_np = v_np.reshape(dim1, dim2)
#     elif dim2 > 1 and dim3 > 1:
#        v_np = v_np.reshape(dim1, dim2, dim3)
#
#     # TODO, test this for 1D, 2D, and 3D arrays, making sure that no copies are done
#
#     return v_np
