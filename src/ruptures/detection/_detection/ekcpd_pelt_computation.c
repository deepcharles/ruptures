#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "kernels.h"
static inline int max_int(int a, int b)
{
    if (a > b)
        return a;
    return b;
}

/**
 * @brief Efficient kernel change point detection
 *
 * @param signal shape (n_samples*n_dims,)
 * @param n_samples number of samples
 * @param n_dims number of dimensions
 * @param beta smoothing parameter
 * @param min_size minimum size of a segment
 * @param kernelDescObj describe the selected kernel
 * @param M_path path matrix of shape (n_samples+1), filled by the function
 */
void ekcpd_pelt_compute(double *signal, int n_samples, int n_dims, double beta, int min_size, void *kernelDescObj, int *M_path)
{
    int t, s;
    int s_min = 0;

    // Allocate memory
    double *D, *S, *M_V, *M_pruning;
    double c_cost, c_cost_sum, c_r, diag_element;

    // Initialize and allocate memory
    // Allocate memory
    D = (double *)malloc((n_samples + 1) * sizeof(double));
    S = (double *)malloc((n_samples + 1) * sizeof(double));
    M_V = (double *)malloc((n_samples + 1) * sizeof(double));
    M_pruning = (double *)malloc((n_samples + 1) * sizeof(double));

    // D, S, M_V and M_path
    for (t = 0; t < (n_samples + 1); t++)
    {
        D[t] = 0.0;
        S[t] = 0.0;
        M_V[t] = 0.0;
        M_path[t] = 0;
        M_pruning[t] = 0.0;
    }

    // for t<2*min_size, there cannot be any change point.
    for (t = 1; t < 2 * min_size; t++)
    {
        diag_element = kernel_value_by_name(&(signal[(t - 1) * n_dims]), &(signal[(t - 1) * n_dims]), n_dims, kernelDescObj);
        D[t] = D[t - 1] + diag_element;

        // Compute S[t-1] = S_{t-1, t}, S[t-2] = S_{t-2, t}, ..., S[0] = S_{0, t}
        // S_{t-1, t} can be computed with S_{t-1, t-1}.
        // S_{t-1, t-1} was stored in S[t-1]
        // S_{t-1, t} will be stored in S[t-1] as well
        c_r = 0.0;
        for (s = t - 1; s >= 0; s--)
        {
            c_r += kernel_value_by_name(&(signal[s * n_dims]), &(signal[(t - 1) * n_dims]), n_dims, kernelDescObj);
            S[s] += 2 * c_r - diag_element;
        }
        c_cost = D[t] - D[0] - S[0] / t;
        M_V[t] = c_cost + beta;
    }

    // Computation loop
    // Handle y_{0..t} = {y_0, ..., y_{t-1}}
    for (t = 2 * min_size; t < (n_samples + 1); t++)
    {
        diag_element = kernel_value_by_name(&(signal[(t - 1) * n_dims]), &(signal[(t - 1) * n_dims]), n_dims, kernelDescObj);
        D[t] = D[t - 1] + diag_element;

        // Compute S[t-1] = S_{t-1, t}, S[t-2] = S_{t-2, t}, ..., S[0] = S_{0, t}
        // S_{t-1, t} can be computed with S_{t-1, t-1}.
        // S_{t-1, t-1} was stored in S[t-1]
        // S_{t-1, t} will be stored in S[t-1] as well
        c_r = 0.0;
        for (s = t - 1; s >= s_min; s--)
        {
            c_r += kernel_value_by_name(&(signal[s * n_dims]), &(signal[(t - 1) * n_dims]), n_dims, kernelDescObj);
            S[s] += 2 * c_r - diag_element;
        }

        // Compute segmentations
        // Store the total cost on y_{0..t} with 0 break points in M_V[t, 0]
        // init
        s = s_min;
        c_cost = D[t] - D[s] - S[s] / (t - s);
        c_cost_sum = M_V[s] + c_cost;
        M_pruning[s] = c_cost_sum;
        c_cost_sum += beta;
        M_V[t] = c_cost_sum;
        M_path[t] = s;
        // search for minimum (penalized) sum of cost
        for (s = max_int(s_min + 1, min_size); s < t - min_size + 1; s++)
        {
            // Compute cost on y_{s..t}
            // D_{s..t} = D_{0..t} - D{0..s} <--> D_{s..t} = D[t] - D[s]
            // S{s..t} has been stored in S[s]
            c_cost = D[t] - D[s] - S[s] / (t - s);
            c_cost_sum = M_V[s] + c_cost;
            M_pruning[s] = c_cost_sum;
            c_cost_sum += beta;
            // Compare to current min
            if (M_V[t] > c_cost_sum)
            {
                M_V[t] = c_cost_sum;
                M_path[t] = s;
            }
        }
        // Pruning
        while ((M_pruning[s_min] >= M_V[t]) && (s_min < t - min_size + 1))
        {
            if (s_min == 0)
            {
                s_min += min_size;
            }
            else
            {
                s_min++;
            }
        }
    }

    // Free memory
    free(D);
    free(S);
    free(M_V);
    free(M_pruning);

    return;
}
