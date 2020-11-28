#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static inline int min_int(int a, int b)
{
    if (a > b)
        return b;
    return a;
}

/**
 * @brief Efficient kernel change point detection
 *
 * @param signal shape (n_samples,)
 * @param n_samples number of samples
 * @param n_dims number of dimensions
 * @param n_bkps number of change points to detect
 * @param min_size minimum size of a segment
 * @param M_path_res path matrix of shape (n_samples+1, n_bkps+1)
 */
void continuous_linear_dynp_c(double *signal, int n_samples, int n_bkps,
                              int min_size, int *M_path)
{
    int t, s, k, length;
    int n_bkps_max;

    // Allocate memory
    double *S, *M_V, *M_c;
    double c_cost, sum_of_costs, alpha_curr, alpha_prev;
    double y_last, y_before_first, y_before_last;

    // Initialize and allocate memory
    // Allocate memory
    S = (double *)malloc((n_samples + 1) * sizeof(double));
    M_c = (double *)malloc((n_samples + 1) * sizeof(double));
    M_V = (double *)malloc((n_samples + 1) * (n_bkps + 1) * sizeof(double));

    // S, M_V and M_path
    for (t = 0; t < (n_samples + 1); t++)
    {
        S[t] = 0.0;
        M_c[t] = 0.0;
        for (k = 0; k < (n_bkps + 1); k++)
        {
            M_V[t * (n_bkps + 1) + k] = 0.0;
            M_path[t * (n_bkps + 1) + k] = 0;
        }
    }

    // Computation loop
    // Handle y_{0..t} = {y_0, ..., y_{t-1}}
    for (t = 2; t < (n_samples + 1); t++)
    {
        y_last = signal[t - 1];
        y_before_last = signal[t - 2];
        // Compute S[t-1] = S_{t-1..t}, S[t-2] = S_{t-2..t},..., S[0] = S_{0..t}
        // S_{t-1..t} can be computed with S_{t-1, t-1}.
        // S_{t-1..t-1} was stored in S[t-1]
        // S_{t-1..t} will be stored in S[t-1] as well
        s = t - 1;
        y_before_first = signal[s - 1];
        S[s] += (t - s) * (y_last - y_before_first);
        for (s = t - 2; s >= 1; s--)
        {
            length = t - s;
            y_before_first = signal[s - 1];
            S[s] += length * (y_last - y_before_first);
            alpha_curr = (y_last - y_before_first) / length;
            alpha_prev = (y_before_last - y_before_first) / (t - 1 - s);
            M_c[s] += 2 * (y_last - y_before_first) * (y_last - y_before_first);
            M_c[s] += (alpha_prev + alpha_curr) * ((length - 1) * (y_last - y_before_first) - (length * (y_before_last - y_before_first))) * (2 * length - 1) / 6.0;
            // M_c[s] += alpha_curr * alpha_curr * sum_of_squared_int(t - s);
            // M_c[s] -= alpha_prev * alpha_prev * sum_of_squared_int(t - s - 1);
            M_c[s] -= 2.0 * (alpha_curr - alpha_prev) * S[s];
            M_c[s] -= 2.0 * alpha_prev * length * (y_last - y_before_first);
        }
        M_c[0] = M_c[1];

        // Compute segmentations
        // Store the total cost on y_{0..t} with 0 break points in M_V[t, 0]
        M_V[t * (n_bkps + 1)] = M_c[0];
        for (s = min_size; s < t - min_size + 1; s++)
        {
            // Compute cost on y_{s..t}
            c_cost = M_c[s];
            // integer division: s / min_size = floor(s / min_size)
            n_bkps_max = min_int(n_bkps, s / min_size);
            for (k = 1; k <= n_bkps_max; k++)
            {
                // With k break points on y_{0..t}, sum cost with (k-1) break
                // points on y_{0..s} and cost on y_{s..t}
                sum_of_costs = M_V[s * (n_bkps + 1) + (k - 1)] + c_cost;
                if (s == k * min_size)
                {
                    // k is the smallest possibility for s in order to have k
                    // break points in y_{0..s}.
                    // It means that y_0, y_1, ..., y_k are break points.
                    M_V[t * (n_bkps + 1) + k] = sum_of_costs;
                    M_path[t * (n_bkps + 1) + k] = s;
                    continue;
                }
                // Compare to current min
                if (M_V[t * (n_bkps + 1) + k] > sum_of_costs)
                {
                    M_V[t * (n_bkps + 1) + k] = sum_of_costs;
                    M_path[t * (n_bkps + 1) + k] = s;
                }
            }
        }
    }

    // Free memory
    free(S);
    free(M_c);
    free(M_V);

    return;
}