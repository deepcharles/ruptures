#include <math.h>
#include <string.h>

#include "kernels.h"

/*************************************
 *
 * Kernels
 *
*************************************/

static inline double linear_kernel(double *x, double *y, int n_dims)
{
	double kernel_value = 0.0;
	for (int dim = 0; dim < n_dims; dim++)
	{
		kernel_value = kernel_value + x[dim] * y[dim];
	}
	return (kernel_value);
}

static inline double gaussian_kernel(double *x, double *y, int n_dims, double gamma)
{
	double squared_distance = 0.0;
	for (int t = 0; t < n_dims; t++)
	{
		squared_distance = squared_distance + (x[t] - y[t]) * (x[t] - y[t]);
	}
	return (exp(-gamma * squared_distance));
}


// Hub function that select proper kernel accoridng kernelObj
double kernel_value_by_name(double *x, double *y, int n_dims, void *kernelObj)
{
	if (strcmp(((KernelL2 *)kernelObj)->pBaseObj->name, LINEAR_KERNEL_NAME) == 0)
	{
		return linear_kernel(x, y, n_dims);
	}
	else if (strcmp(((KernelGaussian *)kernelObj)->pBaseObj->name, GAUSSIAN_KERNEL_NAME) == 0)
	{
		return gaussian_kernel(x, y, n_dims, ((KernelGaussian *)kernelObj)->gamma);
	}
	return 0.0;
}
