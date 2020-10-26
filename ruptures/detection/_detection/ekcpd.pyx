cimport ruptures.detection._detection.ekcpd as ekcpd

from libc.stdlib cimport malloc, free

cimport cython
cimport numpy as cnp
import numpy as np

cpdef cnp.ndarray[cnp.float_t, ndim=1] ekcpd_L2(cnp.ndarray[cnp.float_t, ndim=2] signal, cnp.int n_bkps):

    cdef:
        int n_sample_ = signal.shape[0]
        int n_dim_ = signal.shape[1]
        int n_regimes_ = int(n_bkps)+1
        double gausian_delta = 1.0

    # Allocate and initialize structure for result of c function
    res = <int *>malloc(n_bkps * sizeof(int))
    if not res: raise MemoryError
    cdef _detection.ekcpd.KernelL2 kernelL2Desc
    cdef _detection.ekcpd.KernelGeneric kernelDesc
    kernelDesc.name = LINEAR_KERNEL_NAME
    kernelL2Desc.pBaseObj = &kernelDesc

    # Make it C compatible in terms of memory contiguousness
    cdef double[:, ::1] signal_arr = np.ascontiguousarray(signal)

    try:
        ekcpd.ekcpd(&signal_arr[0, 0], n_sample_, n_dim_, n_bkps, &kernelL2Desc, &res)
    except:
        print("An exception occurred")

    # Transform int* into something understandable by numpy
    cdef int[::1] bkps_flat_arr = <int[:n_bkps]> res

    # Free memory
    free(res)

    return np.asarray(res)