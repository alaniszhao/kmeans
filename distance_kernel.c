#include <immintrin.h>
#include <stdio.h>

#define FMA_ACC256(diff, acc) _mm256_fmadd_ps((diff), (diff), (acc))

/*
  - dist_out should hold 56 floats (8 * 7)
  - each of data_x/data_y/data_z should hold 56 floats
  - cent_x/cent_y/cent_z are single floats (one centroid)
*/
void distance_kernel(float *dist_out,
                          const float *data_x,
                          const float *data_y,
                          const float *data_z,
                          const float cent_x,
                          const float cent_y,
                          const float cent_z)
{
    __m256 R1 = _mm256_setzero_ps();
    __m256 R2 = _mm256_setzero_ps();
    __m256 R3 = _mm256_setzero_ps();
    __m256 R4 = _mm256_setzero_ps();
    __m256 R5 = _mm256_setzero_ps();
    __m256 R6 = _mm256_setzero_ps();
    __m256 R7 = _mm256_setzero_ps();

    __m256 R0 = _mm256_set1_ps(cent_x);

    __m256 R8, R9, R10, R11, R12, R13, R14;
    R8 = _mm256_loadu_ps(data_x +   0);
    R9 = _mm256_loadu_ps(data_x +   8);
    R10 = _mm256_loadu_ps(data_x +  16);
    R11 = _mm256_loadu_ps(data_x +  24);
    R12 = _mm256_loadu_ps(data_x +  32);
    R13 = _mm256_loadu_ps(data_x +  40);
    R14 = _mm256_loadu_ps(data_x +  48);
    R8 = _mm256_sub_ps(R0, R8);
    R9 = _mm256_sub_ps(R0, R9);
    R10 = _mm256_sub_ps(R0, R10);
    R11 = _mm256_sub_ps(R0, R11);
    R12 = _mm256_sub_ps(R0, R12);
    R13 = _mm256_sub_ps(R0, R13);
    R14 = _mm256_sub_ps(R0, R14);
    R1 = FMA_ACC256(R8, R1);
    R2 = FMA_ACC256(R9, R2);
    R3 = FMA_ACC256(R10, R3);
    R4 = FMA_ACC256(R11, R4);
    R5 = FMA_ACC256(R12, R5);
    R6 = FMA_ACC256(R13, R6);
    R7 = FMA_ACC256(R14, R7);

    R0 = _mm256_set1_ps(cent_y);

    R8 = _mm256_loadu_ps(data_y +   0);
    R9 = _mm256_loadu_ps(data_y +   8);
    R10 = _mm256_loadu_ps(data_y +  16);
    R11 = _mm256_loadu_ps(data_y +  24);
    R12 = _mm256_loadu_ps(data_y +  32);
    R13 = _mm256_loadu_ps(data_y +  40);
    R14 = _mm256_loadu_ps(data_y +  48);
    R8 = _mm256_sub_ps(R0, R8);
    R9 = _mm256_sub_ps(R0, R9);
    R10 = _mm256_sub_ps(R0, R10);
    R11 = _mm256_sub_ps(R0, R11);
    R12 = _mm256_sub_ps(R0, R12);
    R13 = _mm256_sub_ps(R0, R13);
    R14 = _mm256_sub_ps(R0, R14);
    R1 = FMA_ACC256(R8, R1);
    R2 = FMA_ACC256(R9, R2);
    R3 = FMA_ACC256(R10, R3);
    R4 = FMA_ACC256(R11, R4);
    R5 = FMA_ACC256(R12, R5);
    R6 = FMA_ACC256(R13, R6);
    R7 = FMA_ACC256(R14, R7);

    R0 = _mm256_set1_ps(cent_z);

    R8 = _mm256_loadu_ps(data_z +   0);
    R9 = _mm256_loadu_ps(data_z +   8);
    R10 = _mm256_loadu_ps(data_z +  16);
    R11 = _mm256_loadu_ps(data_z +  24);
    R12 = _mm256_loadu_ps(data_z +  32);
    R13 = _mm256_loadu_ps(data_z +  40);
    R14 = _mm256_loadu_ps(data_z +  48);
    R8 = _mm256_sub_ps(R0, R8);
    R9 = _mm256_sub_ps(R0, R9);
    R10 = _mm256_sub_ps(R0, R10);
    R11 = _mm256_sub_ps(R0, R11);
    R12 = _mm256_sub_ps(R0, R12);
    R13 = _mm256_sub_ps(R0, R13);
    R14 = _mm256_sub_ps(R0, R14);
    R1 = FMA_ACC256(R8, R1);
    R2 = FMA_ACC256(R9, R2);
    R3 = FMA_ACC256(R10, R3);
    R4 = FMA_ACC256(R11, R4);
    R5 = FMA_ACC256(R12, R5);
    R6 = FMA_ACC256(R13, R6);
    R7 = FMA_ACC256(R14, R7);

    _mm256_storeu_ps(dist_out +   0, R1);
    _mm256_storeu_ps(dist_out +   8, R2);
    _mm256_storeu_ps(dist_out +  16, R3);
    _mm256_storeu_ps(dist_out +  24, R4);
    _mm256_storeu_ps(dist_out +  32, R5);
    _mm256_storeu_ps(dist_out +  40, R6);
    _mm256_storeu_ps(dist_out +  48, R7);
}
