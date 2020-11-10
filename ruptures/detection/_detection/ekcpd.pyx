cimport ruptures.detection._detection.ekcpd as ekcpd

from libc.stdlib cimport malloc, free

cimport cython
import numpy as np

cpdef ekcpd_L2(double[:,:] signal, int n_bkps, int jump, int min_size):

    # Allocate and initialize kernel description
    cdef ekcpd.KernelLinear kernelLinearDesc
    cdef ekcpd.KernelGeneric kernelDesc
    kernelDesc.name = LINEAR_KERNEL_NAME
    kernelLinearDesc.pBaseObj = &kernelDesc

    return ekcpd_core(signal, n_bkps, jump, min_size, &kernelLinearDesc)


cpdef ekcpd_Gaussian(double[:,:] signal, int n_bkps, int jump, int min_size, double gamma):

    # Allocate and initialize kernel description
    cdef ekcpd.KernelGaussian kernelGaussianDesc
    cdef ekcpd.KernelGeneric kernelDesc
    kernelDesc.name = GAUSSIAN_KERNEL_NAME
    kernelGaussianDesc.pBaseObj = &kernelDesc
    kernelGaussianDesc.gamma = gamma

    return ekcpd_core(signal, n_bkps, jump, min_size, &kernelGaussianDesc)


cdef ekcpd_core(double[:,:] signal, int n_bkps, int jump, int min_size, void *kernelDescObj):
    cdef:
        int n_samples = signal.shape[0]
        int n_dims = signal.shape[1]

    # Allocate and initialize structure for result of c function
    q = int(np.ceil(n_samples/jump))
    cdef int[::1] path_matrix_flat = np.empty((n_bkps+1)*(q+1), dtype=np.dtype("i"))
    # Make it C compatible in terms of memory contiguousness
    cdef double[:, ::1] signal_arr = np.ascontiguousarray(signal)
    try:
        ekcpd.ekcpd_compute(&signal_arr[0, 0], n_samples, n_dims, n_bkps, jump, min_size, kernelDescObj, &path_matrix_flat[0])
    except:
        print("An exception occurred.")

    return np.asarray(path_matrix_flat)