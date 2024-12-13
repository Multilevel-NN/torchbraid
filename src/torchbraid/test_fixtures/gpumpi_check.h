#include <stdio.h>
#include "mpi.h"
#include "mpi-ext.h" /* Needed for CUDA-aware check */

/* Code from: https://www.open-mpi.org/faq/?category=runcuda#mpi-cuda-aware-support */
int gpumpi_direct_support()
{
    int has_compile_time = 0;
    int has_run_time = 0;

    printf("Check For GPU-Direct Support\n");

    printf("-- compile time: ");
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
    printf("This MPI library has CUDA-aware support.\n");
    has_compile_time = 1;
#elif defined(MPIX_CUDA_AWARE_SUPPORT) && !MPIX_CUDA_AWARE_SUPPORT
    printf("This MPI library does NOT have CUDA-aware support.\n");
#else
    printf("This MPI library cannot determine if there is CUDA-aware support.\n");
#endif /* MPIX_CUDA_AWARE_SUPPORT */

    printf("--run time:");
#if defined(MPIX_CUDA_AWARE_SUPPORT)
    if (1 == MPIX_Query_cuda_support()) {
        printf("This MPI library has CUDA-aware support.\n");
        has_run_time = 1;
    } else {
        printf("This MPI library does not have CUDA-aware support.\n");
    }
#else /* !defined(MPIX_CUDA_AWARE_SUPPORT) */
    printf("This MPI library cannot determine if there is CUDA-aware support.\n");
#endif /* MPIX_CUDA_AWARE_SUPPORT */

    printf("\n");

    return has_compile_time && has_run_time;
}
