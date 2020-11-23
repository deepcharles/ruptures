#include <math.h>
#include <string.h>

#include "kernels.h"

/*************************************
 *
 * Utils
 *
*************************************/

static inline float min_f(float a, float b)
{
	if (a > b)
		return b;
	return a;
}

static inline float max_f(float a, float b)
{
	if (a > b)
		return a;
	return b;
}

float clip(float n, float lower, float upper)
{
	return max_f(lower, min_f(n, upper));
}

/*************************************
 *
 * Kernels
 *
*************************************/

static inline double linear_kernel(double *x, double *y, int n_dims)
{
	double kernel_value = 0.0;
	int dim;
	for (dim = 0; dim < n_dims; dim++)
	{
		kernel_value = kernel_value + x[dim] * y[dim];
	}
	return (kernel_value);
}

static inline double gaussian_kernel(double *x, double *y, int n_dims, double gamma)
{
	double squared_distance = 0.0;
	int t;
	for (t = 0; t < n_dims; t++)
	{
		squared_distance = squared_distance + (x[t] - y[t]) * (x[t] - y[t]);
	}
	// clipping to avoid exp under/overflow
	return exp(-clip(gamma * squared_distance, 0.01, 100));
}

static inline double cosine_similarity(double *x, double *y, int n_dims)
{
    double dot = 0.0, denom_x = 0.0, denom_y = 0.0 ;
	int i;
     for(i = 0; i < n_dims; i++) {
        dot += x[i] * y[i] ;
        denom_x += x[i] * x[i] ;
        denom_y += y[i] * y[i] ;
    }
    return dot / (sqrt(denom_x) * sqrt(denom_y)) ;
}



// Hub function that select proper kernel accoridng kernelObj
double kernel_value_by_name(double *x, double *y, int n_dims, void *kernelObj)
{
	if (strcmp(((KernelLinear *)kernelObj)->pBaseObj->name, LINEAR_KERNEL_NAME) == 0)
	{
		return linear_kernel(x, y, n_dims);
	}
	else if (strcmp(((KernelGaussian *)kernelObj)->pBaseObj->name, GAUSSIAN_KERNEL_NAME) == 0)
	{
		return gaussian_kernel(x, y, n_dims, ((KernelGaussian *)kernelObj)->gamma);
	}
	else if (strcmp(((KernelCosine *)kernelObj)->pBaseObj->name, COSINE_KERNEL_NAME) == 0)
	{
		return cosine_similarity(x, y, n_dims);
	}
	return 0.0;
}
