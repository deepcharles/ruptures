#define MAX_KERNEL_NAME_LENGTH 10
#define LINEAR_KERNEL_NAME "linear\0"
#define GAUSSIAN_KERNEL_NAME "gaussian\0"

typedef struct KernelGeneric {
    char *name;
} KernelGeneric;

typedef struct KernelL2 {
    KernelGeneric *pBaseObj;
} KernelL2;

typedef struct KernelGaussian {
    KernelGeneric *pBaseObj;
    double gamma;
} KernelGaussian;

double kernel_value_by_name(double *x, double *y, int n_dims, void *kernelObj);
