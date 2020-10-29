#include <string.h>
#include <stdlib.h>

#include "kernels.h"

/**
 * @brief Efficient kernel change point detection
 *
 * @param signal shape (n_samples*n_dims,)
 * @param n_samples number of samples
 * @param n_dims number of dimensions
 * @param n_bkps number of change points to detect
 * @param kernelDescObj describe the selected kernel
 * @param M_path path matrix of shape ((n_samples+1)*(n_bkps+1),) that will
 * hold the results, must be allocated beforehand.
 */
void ekcpd_compute(double *signal, int n_samples, int n_dims, int n_bkps, void *kernelDescObj, int *M_path)
{
    int i, j, t, s, k;
    int n_bkps_max;

    // Allocate memory
    double *D, *S, *M_V;
    double c_cost, c_cost_sum, c_r;

    D = (double *)malloc((n_samples + 1) * sizeof(double));
    S = (double *)malloc((n_samples + 1) * sizeof(double));
    M_V = (double *)malloc((n_bkps + 1) * (n_samples + 1) * sizeof(double));

    // Initialize
    c_cost = 0;
    c_cost_sum = 0;
    c_r = 0;
    // D and S
    for (i = 0; i < (n_samples + 1); i++)
    {
        D[i] = 0.0;
        S[i] = 0.0;
    }
    // M_V and M_path
    for (i = 0; i < (n_samples + 1); i++)
    {
        for (j = 0; j < (n_bkps + 1); j++)
        {
            M_V[i * (n_bkps + 1) + j] = 0.0;
            M_path[i * (n_bkps + 1) + j] = 0;
        }
    }

    // Computation loop
    // Handle y_{0..t} = {y_0, ..., y_{t-1}}
    for (t = 1; t < (n_samples + 1); t++)
    {
        // Compute D[t], D[t] = D_{0..t}
        D[t] = D[t - 1] + kernel_value_by_name(&(signal[(t - 1) * n_dims]), &(signal[(t - 1) * n_dims]), n_dims, kernelDescObj);

        // Compute S[t-1] = S_{t-1, t}, S[t-2] = S_{t-2, t}, ..., S[0] = S_{0, t}
        // S_{t-1, t} Can be computed with S_{t-1, t-1}.
        // S_{t-1, t-1} was stored in S[t-1]
        // S_{t-1, t} will be stored in S[t-1] as well
        c_r = 0.0;
        for (s = t - 1; s >= 0; s--)
        {
            c_r += kernel_value_by_name(&(signal[s * n_dims]), &(signal[(t - 1) * n_dims]), n_dims, kernelDescObj);
            S[s] += 2 * c_r - kernel_value_by_name(&(signal[(t - 1) * n_dims]), &(signal[(t - 1) * n_dims]), n_dims, kernelDescObj);
        }

        // Compute segmentation
        // Store the total cost on y_{0..t} with 0 break points in M_V[t, 0]
        M_V[t * (n_bkps + 1)] = D[t] - S[0] / t;
        for (s = 1; s <= (t - 1); s++)
        {
            // Compute cost on y_{s..t}
            // D_{s..t} = D_{0..t} - D{0..s} <--> D_{s..t} = D[t] - D[s]
            // S{s..t} has been stored in S[s]
            c_cost = D[t] - D[s] - S[s] / (t - s);

            // Maximum number of break points on the segment y_{0..s}
            if (n_bkps < s)
            {
                n_bkps_max = n_bkps;
            }
            else
            {
                n_bkps_max = s;
            }

            for (k = 1; k <= n_bkps_max; k++)
            {
                // With k break points on y_{0..t}, sum cost with (k-1) break points on y_{0..s} and cost on y_{s..t}
                c_cost_sum = M_V[s * (n_bkps + 1) + (k - 1)] + c_cost;
                if (s == k)
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
