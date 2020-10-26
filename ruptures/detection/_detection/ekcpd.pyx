cimport ruptures.detection._detection.ekcpd as ekcpd

from libc.stdlib cimport malloc, free

cimport cython
cimport numpy as cnp
import numpy as np

cpdef cnp.ndarray[cnp.float_t, ndim=1] ekcpd_L2(cnp.ndarray[cnp.float_t, ndim=2] signal, cnp.int n_bkps):

    # Allocate and initialize kernel description
    cdef ekcpd.KernelLinear kernelLinearDesc
    cdef ekcpd.KernelGeneric kernelDesc
    kernelDesc.name = LINEAR_KERNEL_NAME
    kernelLinearDesc.pBaseObj = &kernelDesc

    return ekcpd_core(signal, n_bkps, &kernelLinearDesc)


cpdef cnp.ndarray[cnp.float_t, ndim=1] ekcpd_Gaussian(cnp.ndarray[cnp.float_t, ndim=2] signal, cnp.int n_bkps, cnp.float_t gamma):

    # Allocate and initialize kernel description
    cdef ekcpd.KernelGaussian kernelGaussianDesc
    cdef ekcpd.KernelGeneric kernelDesc
    kernelDesc.name = GAUSSIAN_KERNEL_NAME
    kernelGaussianDesc.pBaseObj = &kernelDesc
    kernelGaussianDesc.gamma = gamma

    return ekcpd_core(signal, n_bkps, &kernelGaussianDesc)

cdef cnp.ndarray[cnp.float_t, ndim=1] ekcpd_core(cnp.ndarray[cnp.float_t, ndim=2] signal, cnp.int n_bkps, void *kernelDescObj):
    cdef:
        int n_sample_ = signal.shape[0]
        int n_dim_ = signal.shape[1]

    # Allocate and initialize structure for result of c function
    res = <int *>malloc(n_bkps * sizeof(int))
    # Make it C compatible in terms of memory contiguousness
    cdef double[:, ::1] signal_arr = np.ascontiguousarray(signal)
    try:
        ekcpd.ekcpd_compute(&signal_arr[0, 0], n_sample_, n_dim_, n_bkps, kernelDescObj, &res)
    except:
        print("An exception occurred")

    # Transform int* into something understandable by numpy
    cdef int[::1] bkps_flat_arr = <int[:n_bkps]> res

    # Free memory
    free(res)

    return np.asarray(bkps_flat_arr)