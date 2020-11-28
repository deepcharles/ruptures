cdef extern from "continuous_linear_dynp_c.h":
    void continuous_linear_dynp_c(double *signal, int n_samples, int n_bkps, int min_size, int *M_path)