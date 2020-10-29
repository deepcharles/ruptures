#include <stdio.h>

void convert_path_matrix_c(int *path_matrix, int n_bkps, int n_samples, int n_bkps_max, int *bkps_list)
{
    bkps_list[n_bkps] = n_samples;
    int k = 0;
    while (k++ < n_bkps)
    {
        bkps_list[n_bkps - k] = path_matrix[bkps_list[n_bkps - k + 1] * (n_bkps_max + 1) + (n_bkps - k + 1)];
    }
    return;
}
