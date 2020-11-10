#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "kernels.h"

static inline int min(int a, int b)
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
void ekcpd_compute(double *signal, int n_samples, int n_dims, int n_bkps, int jump, int min_size, void *kernelDescObj, int *M_path_res, int *q_res)
{
    int i, j, t, s, k;
    int q, q_t, q_s, q_s_max;
    // Allocate memory
    double *D, *S_off_diag, *S_diag, *M_V;
    double c_cost, c_cost_sum, c_r;
    double d_current, acc, c_current, v_current;
    int *M_path = M_path_res;

    // Initialize and allocate memory
    q = (int)ceil((float)n_samples / (float)jump); // Number of eligible break points
    // Allocate memory
    D = (double *)malloc((q + 1) * sizeof(double));
    S_off_diag = (double *)malloc((q + 1) * sizeof(double));
    S_diag = (double *)malloc((q + 1) * sizeof(double));
    M_V = (double *)malloc((q + 1) * (n_bkps + 1) * sizeof(double));

    // D, S_off_diag, S_diag
    for (i = 0; i < (q + 1); i++)
    {
        D[i] = 0.0;
        S_off_diag[i] = 0.0;
        S_diag[i] = 0.0;
    }
    // M_V and M_path
    for (i = 0; i < (q + 1); i++)
    {
        for (j = 0; j < (n_bkps + 1); j++)
        {
            M_V[i * (n_bkps + 1) + j] = 0.0;
            M_path[i * (n_bkps + 1) + j] = 0;
        }
    }
    d_current = 0.0;

    // Computation loop
    // Handle jumps iteratively : q_t = [1, ..., q]
    for (q_t = 1; q_t < (q + 1); q_t++)
    {
        // Handle all signal points within the interval [(q_t - 1) * jump + 1, min(q_t * jump, n_samples)]
        S_off_diag[q_t] = S_off_diag[q_t - 1];
        for (t = (q_t - 1) * jump + 1; t < min(q_t * jump, n_samples) + 1; t++)
        {
            d_current += kernel_value_by_name(&(signal[(t - 1) * n_dims]), &(signal[(t - 1) * n_dims]), n_dims, kernelDescObj);
            // Update S_off_diag[1], S_off_diag[2], ..., S_off_diag[q_t]
            // S_off_diag[qs] = S_{0 .. q_s * jump, 0 .. q_t * jump}
            acc = 0.0;
            for (q_s = 1; q_s < q_t; q_s++)
            {
                // Handle all signal points within the interval [(q_s - 1) * jump, min(q_t * jump, n_samples)]
                for (s = (q_s - 1) * jump + 1; s < q_s * jump + 1; s++)
                {
                    acc += kernel_value_by_name(&(signal[(s - 1) * n_dims]), &(signal[(t - 1) * n_dims]), n_dims, kernelDescObj);
                }
                S_off_diag[q_s] += acc;
            }
            for (s = (q_t - 1) * jump + 1; s < t; s++)
            {
                acc += kernel_value_by_name(&(signal[(s - 1) * n_dims]), &(signal[(t - 1) * n_dims]), n_dims, kernelDescObj);
            }
            S_off_diag[q_t] += 2 * acc + kernel_value_by_name(&(signal[(t - 1) * n_dims]), &(signal[(t - 1) * n_dims]), n_dims, kernelDescObj);
        }
        // D[q_t] = D_{0 .. q_t * jump}
        D[q_t] = d_current;

        // S_diag[q_t] = S_{0 .. q_t * jump, 0 .. q_t * jump}
        S_diag[q_t] = S_off_diag[q_t];

        // Compute segmentations
        // M_V[q_t, 0] = c(y_{0 .. q_t*jump})
        M_V[q_t * (n_bkps + 1)] = D[q_t] - S_diag[q_t] / (q_t * jump);
        q_s_max = q_t - (min_size - 1) / jump - 1; // integer division
        for (q_s = 1; q_s < q_s_max + 1; q_s++)
        {
            c_current = D[q_t] - D[q_s] - (S_diag[q_t] + S_diag[q_s] - 2 * S_off_diag[q_s]) / ((q_t - q_s) * jump); // = c(y_{q_s*jump .. q_t*jump})
            for (k = 1; k < min(n_bkps, q_s) + 1; k++)
            {
                v_current = M_V[q_s * (n_bkps + 1) + (k - 1)] + c_current; // = V_{k-1}(y_{0 .. q_s*jump}) + c(y_{q_s*jump .. q_t*jump})
                if (q_s == k)
                {
                    M_V[q_t * (n_bkps + 1) + k] = v_current;
                    M_path[q_t * (n_bkps + 1) + k] = q_s;
                }
                else
                {
                    if (M_V[q_t * (n_bkps + 1) + k] > v_current)
                    {
                        M_V[q_t * (n_bkps + 1) + k] = v_current;
                        M_path[q_t * (n_bkps + 1) + k] = q_s;
                    }
                }
            }
        }
    }

    // Free memory
    free(D);
    free(S_off_diag);
    free(S_diag);
    free(M_V);

    return;
}
