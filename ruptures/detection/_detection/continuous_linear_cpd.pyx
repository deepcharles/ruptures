cimport ruptures.detection._detection.continuous_linear_cpd as clin

from libc.stdlib cimport malloc, free

cimport cython
import numpy as np


cpdef continuous_linear_dynp(double[:] signal, int n_bkps, int min_size):
    cdef:
        int n_samples = signal.shape[0]

    # Allocate and initialize structure for result of c function
    cdef int[::1] path_matrix_flat = np.empty((n_bkps+1)*(n_samples+1), dtype=np.dtype("i"))
    # Make it C compatible in terms of memory contiguousness
    cdef double[::1] signal_arr = np.ascontiguousarray(signal)
    try:
        clin.continuous_linear_dynp_c(&signal_arr[0], n_samples, n_bkps, min_size, &path_matrix_flat[0])
    except:
        print("An exception occurred.")

    return np.asarray(path_matrix_flat)