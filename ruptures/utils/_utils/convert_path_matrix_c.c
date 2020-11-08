#include <stdio.h>

void convert_path_matrix_c(int *path_matrix, int n_bkps, int n_eligible_bkps, int n_bkps_max, int jump, int *bkps_list)
{
    bkps_list[n_bkps] = n_eligible_bkps;
    int k = 0;
    printf("Before while\n");
    while (k++ < n_bkps)
    {
        printf("k is %d\n", k);
        printf("%d\n", bkps_list[n_bkps - k + 1]);
        bkps_list[n_bkps - k] = path_matrix[bkps_list[n_bkps - k + 1] * (n_bkps_max + 1) + (n_bkps - k + 1)];
    }
    for (k = 0 ; k < n_bkps + 1 ; k++)
    {
        bkps_list[k] = bkps_list[k] * jump;
    }
    return;
}
