#include <math.h>

void convert_path_matrix_c(int *path_matrix, int n_bkps, int n_samples, int n_bkps_max, int jump, int *bkps_list)
{
    int q = (int)ceil((float)n_samples / (float)jump);
    bkps_list[n_bkps] = q;
    int k = 0;
    while (k++ < n_bkps)
    {
        bkps_list[n_bkps - k] = path_matrix[bkps_list[n_bkps - k + 1] * (n_bkps_max + 1) + (n_bkps - k + 1)];
    }
    for (k = 0 ; k < n_bkps + 1 ; k++)
    {
        bkps_list[k] = bkps_list[k] * jump;
    }
    bkps_list[n_bkps] = n_samples;
    return;
}
