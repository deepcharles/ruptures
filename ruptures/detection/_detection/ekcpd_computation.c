#include <string.h>
#include <stdlib.h>
#include <math.h>

// To be deleted
#include <stdio.h>

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
 * @param M_path_res path matrix of shape ?? that will
 * hold the results.
 * @param q_res Number of eligible break points, will be affected by the function
 */
void ekcpd_compute(double *signal, int n_samples, int n_dims, int n_bkps, int jump, int min_size, void *kernelDescObj, int *M_path_res, int *q_res)
{
    int i, j, t, s, k; // To be checked
    int n_bkps_max; // To be checked
    int q, q_t, q_s, q_s_max;
    // Allocate memory
    double *D, *S_off_diag, *S_diag, *M_V;
    double c_cost, c_cost_sum, c_r;
    double d_current, acc, c_current, v_current;
    int * M_path = M_path_res;

    // Debug
    printf("In ekcpd_compute with new signature\n");
    printf("%d %d\n", jump, min_size);

    // Initialize and allocate memory
    q = (int)ceil(n_samples/jump);  // Number of eligible break points
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
    // To be checked
    c_cost = 0;
    c_cost_sum = 0;
    c_r = 0;

    // Debug
    printf("Initialization done\n");
    printf("q is %d\n", q);
    printf("%f %f %f %f\n", signal[0], signal[1], signal[2], signal[n_dims]);

    // Computation loop
    // Handle jumps iteratively : q_t = [1, ..., q]
    for (q_t = 1; q_t < (q + 1) ; q_t++)
    {
        // Handle all signal points within the interval [(q_t - 1) * jump, min(q_t * jump, n_samples)]
        for (t = (q_t - 1) * jump  + 1; t < min(q_t * jump, n_samples) + 1; t++)
        {
            d_current += kernel_value_by_name(&(signal[(t - 1) * n_dims]), &(signal[(t - 1) * n_dims]), n_dims, kernelDescObj);
            // Update S_off_diag[1], S_off_diag[2], ..., S_off_diag[q_t]
            // S_off_diag[qs] = S_{0 .. q_s * jump, 0 .. q_t * jump}
            acc = 0;
            for (q_s = 1 ; q_s < q_t + 1 ; q_s++)
            {
                // Handle all signal points within the interval [(q_s - 1) * jump, min(q_t * jump, n_samples)]
                for (s = (q_s - 1) * jump + 1 ; s < min(q_s * jump, t) + 1 ; s++)
                {
                    acc += kernel_value_by_name(&(signal[(s - 1) * n_dims]), &(signal[(t - 1) * n_dims]), n_dims, kernelDescObj);
                }
                S_off_diag[q_s] +=  acc;
            }
        }
        // D[q_t] = D_{0 .. q_t * jump}
        D[q_t] = d_current;
        // S_diag[q_t] = S_{0 .. q_t * jump, 0 .. q_t * jump}
        S_diag[q_t] = S_off_diag[q_t];

        // debug
        printf("q_t is %d and D[q_t] is %f\n", q_t, D[q_t]);
        //printf("S_diag[q_t] is %f\n", S_diag[q_t]);

        // Compute segmentations
        // M_V[q_t, 0] = c(y_{0 .. q_t*jump})
        M_V[q_t * (n_bkps + 1)] = D[q_t] - S_diag[q_t] / (q_t*jump);
        q_s_max = q_t - (int)floor((min_size - 1)/jump) - 1;
        for (q_s = 1 ; q_s < q_s_max + 1  ; q_s++)
        {
            c_current = D[q_t] - D[q_s] - (S_diag[q_t] + S_diag[q_s] - 2*S_off_diag[q_s])/((q_t-q_s)*jump); // = c(y_{q_s*jump .. q_t*jump})
            //printf("c_current %f\n", c_current);
            for (k = 1 ; k < min(n_bkps, q_s) + 1; k++)
            {
                //printf("k %d is and max condition is %d\n", k, min(n_bkps, q_s));
                v_current = M_V[q_s * (n_bkps + 1) + (k-1)] + c_current; // = V_{k-1}(y_{0 .. q_s*jump}) + c(y_{q_s*jump .. q_t*jump})
                if (q_s == k)
                {
                    //printf("in case q_s == k, %d\n", q_s);
                    M_V[q_t * (n_bkps + 1) + k] = v_current;
                    M_path[q_t * (n_bkps + 1) + k] = q_s;
                }
                else
                {
                    if (M_V[q_t * (n_bkps + 1) + k] > v_current)
                    {
                        //printf("Updating with q_s and k, %d, %d\n", q_s, k);
                        M_V[q_t * (n_bkps + 1) + k] = v_current;
                        M_path[q_t * (n_bkps + 1) + k] = q_s;
                    }
                }
            }
        }
    }
    // Debug
    printf("Debugging\n");
    printf("%d\n", M_path[0]);
    printf("%d\n", M_path[2]);
    printf("%d\n", M_path[(q + 1) * (n_bkps + 1) -2]);
    printf("%d\n", M_path[(q + 1) * (n_bkps + 1) -1]);

    // Return results
    //*M_path_res = &M_path[0];
    *q_res = q;


    // Debug
    printf("Computation done\n");

    // Free memory
    free(D);
    free(S_off_diag);
    free(S_diag);
    free(M_V);

    return;
}
