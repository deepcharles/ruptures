#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "kernels.h"

static inline int int_min(int a, int b)
{
    if (a > b)
        return b;
    return a;
}

/**
 * @brief Efficient kernel change point detection
 *
 * @param signal shape (n_samples*n_dims,)
 * @param n_samples number of samples
 * @param n_dims number of dimensions
 * @param n_bkps number of change points to detect
 * @param jump index jump while scanning through the signal data
 * @param min_size minimum size of a segment
 * @param kernelDescObj describe the selected kernel
 * @param M_path_res path matrix of shape (q+1, n_bkps+1) where q = ceil(q/jump)
 */
void ekcpd_compute(double *signal, int n_samples, int n_dims, int n_bkps, int min_size, void *kernelDescObj, int *M_path)
{
    int t, s, k;
    int n_bkps_max;

    // Allocate memory
    double *D, *S, *M_V;
    double c_cost, c_cost_sum, c_r, diag_element;

    // Initialize and allocate memory
    // Allocate memory
    D = (double *)malloc((n_samples + 1) * sizeof(double));
    S = (double *)malloc((n_samples + 1) * sizeof(double));
    M_V = (double *)malloc((n_samples + 1) * (n_bkps + 1) * sizeof(double));

    // D, S, M_V and M_path
    for (t = 0; t < (n_samples + 1); t++)
    {
        D[t] = 0.0;
        S[t] = 0.0;
        for (k = 0; k < (n_bkps + 1); k++)
        {
            M_V[t * (n_bkps + 1) + k] = 0.0;
            M_path[t * (n_bkps + 1) + k] = 0;
        }
    }

    // Computation loop
    // Handle y_{0..t} = {y_0, ..., y_{t-1}}
    for (t = 1; t < (n_samples + 1); t++)
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

        // Compute segmentations
        // Store the total cost on y_{0..t} with 0 break points in M_V[t, 0]
        M_V[t * (n_bkps + 1)] = D[t] - S[0] / t;
        for (s = min_size; s < t - min_size + 1; s++)
        {
            // Compute cost on y_{s..t}
            // D_{s..t} = D_{0..t} - D{0..s} <--> D_{s..t} = D[t] - D[s]
            // S{s..t} has been stored in S[s]
            c_cost = D[t] - D[s] - S[s] / (t - s);
            n_bkps_max = int_min(n_bkps, s / min_size); // integer division: s / min_size = floor(s / min_size)
            for (k = 1; k <= n_bkps_max; k++)
            {
                // With k break points on y_{0..t}, sum cost with (k-1) break points on y_{0..s} and cost on y_{s..t}
                c_cost_sum = M_V[s * (n_bkps + 1) + (k - 1)] + c_cost;
                if (s == k * min_size)
                {
                    // k is the smallest possibility for s in order to have k break points in y_{0..s}.
                    // It means that y_0, y_1, ..., y_k are break points.
                    M_V[t * (n_bkps + 1) + k] = c_cost_sum;
                    M_path[t * (n_bkps + 1) + k] = s;
                    continue;
                }
                // Compare to current min
                if (M_V[t * (n_bkps + 1) + k] > c_cost_sum)
                {
                    M_V[t * (n_bkps + 1) + k] = c_cost_sum;
                    M_path[t * (n_bkps + 1) + k] = s;
                }
            }
        }
    }

    // Free memory
    free(D);
    free(S);
    free(M_V);

    return;
}
