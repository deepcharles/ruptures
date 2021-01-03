cimport ruptures.utils._utils.convert_path_matrix as convert_pm

cimport cython
import numpy as np

cpdef from_path_matrix_to_bkps_list(int[:] path_matrix_flat, int n_bkps, int n_samples, int n_bkps_max, int jump):
    # Init bkps_list array
    cdef int[::1] bkps_list = np.empty((n_bkps+1), dtype=np.dtype("i"))
    try:
        convert_pm.convert_path_matrix_c(&path_matrix_flat[0], n_bkps, n_samples, n_bkps_max, jump, &bkps_list[0])
    except:
        print("An exception occurred.")
    return np.asarray(bkps_list).tolist()