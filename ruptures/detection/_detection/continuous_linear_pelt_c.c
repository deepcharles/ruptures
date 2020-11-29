#include <stdlib.h>
#include <math.h>

static inline int max_int(int a, int b)
{
    if (a > b)
        return a;
    return b;
}

/**
 * @brief Optimal 1D linear spline smoothing (penalized number of knots).
 *
 * @param signal shape (n_samples,)
 * @param n_samples number of samples
 * @param beta smoothing parameter
 * @param min_size minimum size of a segment
 * @param M_path path matrix of shape (n_samples+1), filled by the function
 */
void continuous_linear_pelt_c(double *signal, int n_samples, double beta, int min_size, int *M_path)
{
    int t, s, length;
    int s_min = 0;

    // Allocate memory
    double *S, *M_V, *M_pruning, *M_c;
    double c_cost, sum_of_costs, alpha_curr, alpha_prev;
    double y_last, y_before_first, y_before_last;

    // Initialize and allocate memory
    // Allocate memory
    S = (double *)malloc((n_samples + 1) * sizeof(double));
    M_c = (double *)malloc((n_samples + 1) * sizeof(double));
    M_V = (double *)malloc((n_samples + 1) * sizeof(double));
    M_pruning = (double *)malloc((n_samples + 1) * sizeof(double));

    // init arrays
    for (t = 0; t < (n_samples + 1); t++)
    {
        S[t] = 0.0;
        M_c[t] = 0.0;
        M_V[t] = 0.0;
        M_path[t] = 0.0;
        M_pruning[t] = 0.0;
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
        for (s = t - 2; s >= max_int(s_min, 1); s--)
        {
            length = t - s;
            y_before_first = signal[s - 1];
            S[s] += length * (y_last - y_before_first);
            alpha_curr = (y_last - y_before_first) / length;
            alpha_prev = (y_before_last - y_before_first) / (t - 1 - s);
            M_c[s] += 2 * (y_last - y_before_first) * (y_last - y_before_first);
            M_c[s] += (alpha_prev + alpha_curr) * ((length - 1) * (y_last - y_before_first) - (length * (y_before_last - y_before_first))) * (2 * length - 1) / 6.0;
            M_c[s] -= 2.0 * (alpha_curr - alpha_prev) * S[s];
            M_c[s] -= 2.0 * alpha_prev * length * (y_last - y_before_first);
        }
        M_c[0] = M_c[1];

        // Compute segmentations
        // Store the total cost on y_{0..t} with 0 break points in M_V[t, 0]
        // init
        s = s_min;
        c_cost = M_c[s];
        sum_of_costs = M_V[s] + c_cost;
        M_pruning[s] = sum_of_costs;
        sum_of_costs += beta;
        M_V[t] = sum_of_costs;
        M_path[t] = s;
        // search for minimum (penalized) sum of cost
        for (s = s_min + 1; s < t - min_size + 1; s++)
        {
            // Compute cost on y_{s..t}
            // D_{s..t} = D_{0..t} - D{0..s} <--> D_{s..t} = D[t] - D[s]
            // S{s..t} has been stored in S[s]
            c_cost = M_c[s];
            sum_of_costs = M_V[s] + c_cost;
            M_pruning[s] = sum_of_costs;
            sum_of_costs += beta;
            // Compare to current min
            if (M_V[t] > sum_of_costs)
            {
                M_V[t] = sum_of_costs;
                M_path[t] = s;
            }
        }
        // Pruning
        while ((M_pruning[s_min] >= M_V[t]) && (s_min < t - min_size + 1))
        {
            s_min++;
        }
    }
    // Free memory
    free(S);
    free(M_c);
    free(M_pruning);
    free(M_V);

    return;
}
