cdef extern from "kernels.h":
    ctypedef struct KernelGeneric:
        char *name
    ctypedef struct KernelL2:
        KernelGeneric *pBaseObj
    ctypedef struct KernelGaussian:
        KernelGeneric *pBaseObj
        double delta
    cdef char *LINEAR_KERNEL_NAME
    cdef char *GAUSSIAN_KERNEL_NAME
    double kernel_value_by_name(double *x, double *y, int n_dims, void *kernelObj)

cdef extern from "ekcpd_computation.h":
    void ekcpd_compute(double *signal, int n_samples, int n_dims, int n_bkps, void *kernelDescObj, int **res)




