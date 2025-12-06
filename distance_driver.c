#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>

#define MAX_FREQ 3.4
#define BASE_FREQ 2.4
#define BLOCK_SIZE 56

static __inline__ unsigned long long rdtsc(void) {
    unsigned hi, lo;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return ((unsigned long long)lo) | (((unsigned long long)hi) << 32);
}

void distance_kernel(float *dist_out,
                     const float *data_x, const float *data_y, const float *data_z,
                     const float cent_x, const float cent_y, const float cent_z);

void distance_naive_block(float *dist_out,
                          const float *data_x, const float *data_y, const float *data_z,
                          const float cent_x, const float cent_y, const float cent_z)
{
    for (int i = 0; i < BLOCK_SIZE; ++i) {
        float dx = cent_x - data_x[i];
        float dy = cent_y - data_y[i];
        float dz = cent_z - data_z[i];
        dist_out[i] = dx*dx + dy*dy + dz*dz;
    }
}

int main() {
    int sizes[] = {56, 56*10, 56*20, 56*40, 56*80, 56*160, 56*320, 56*3200};
    const int n_sizes = sizeof(sizes) / sizeof(sizes[0]);
    const int iterations = 10000;

    for (int s = 0; s < n_sizes; ++s) {
        int N = sizes[s];

        float *data_x, *data_y, *data_z;
        float *cent_x, *cent_y, *cent_z;
        float *dist_simd, *dist_ref;

        posix_memalign((void**)&data_x, 64, N * sizeof(float));
        posix_memalign((void**)&data_y, 64, N * sizeof(float));
        posix_memalign((void**)&data_z, 64, N * sizeof(float));
        posix_memalign((void**)&cent_x, 64, N * sizeof(float));
        posix_memalign((void**)&cent_y, 64, N * sizeof(float));
        posix_memalign((void**)&cent_z, 64, N * sizeof(float));
        posix_memalign((void**)&dist_simd, 64, N * sizeof(float));
        posix_memalign((void**)&dist_ref, 64, N * sizeof(float));

        for (int i = 0; i < N; ++i) {
            data_x[i] = ((float)rand()) / RAND_MAX;
            data_y[i] = ((float)rand()) / RAND_MAX;
            data_z[i] = ((float)rand()) / RAND_MAX;
            cent_x[i] = ((float)rand()) / RAND_MAX;
            cent_y[i] = ((float)rand()) / RAND_MAX;
            cent_z[i] = ((float)rand()) / RAND_MAX;
            dist_simd[i] = 0.0f;
            dist_ref[i]  = 0.0f;
        }

        int blocks = N / BLOCK_SIZE;

        if (s == 0) {
            for (int b = 0; b < blocks; ++b) {
                int offset = b * BLOCK_SIZE;
                distance_naive_block(dist_ref + offset,
                                     data_x + offset, data_y + offset, data_z + offset,
                                     cent_x[offset], cent_y[offset], cent_z[offset]);
                distance_kernel(dist_simd + offset,
                                data_x + offset, data_y + offset, data_z + offset,
                                cent_x[offset], cent_y[offset], cent_z[offset]);
            }

            int correct = 1;
            for (int i = 0; i < N; ++i) {
                if (fabsf(dist_simd[i] - dist_ref[i]) > 1e-5f) {
                    printf("Mismatch at %d: SIMD=%f REF=%f\n", i, dist_simd[i], dist_ref[i]);
                    correct = 0;
                    break;
                }
            }
            printf("correct: %s\n", correct ? "YES" : "NO");
            printf("size,throughput\n");
        }

        unsigned long long t0 = rdtsc();
        for (int it = 0; it < iterations; ++it) {
            for (int b = 0; b < blocks; ++b) {
                int offset = b * BLOCK_SIZE;
                distance_kernel(dist_simd + offset,
                                data_x + offset, data_y + offset, data_z + offset,
                                cent_x[offset], cent_y[offset], cent_z[offset]);
            }
        }
        unsigned long long t1 = rdtsc();

        double total_flops = (double) N * 9.0 * (double) iterations; // 3 sub + 3 fma (mul+add)
        double cycles = (double)(t1 - t0);
        cycles = (double)(MAX_FREQ/BASE_FREQ)*cycles;
        double throughput = total_flops / cycles;

        printf("%d,%.6f\n", N, throughput);

        free(data_x);
        free(data_y);
        free(data_z);
        free(cent_x);
        free(cent_y);
        free(cent_z);
        free(dist_simd);
        free(dist_ref);
    }

    return 0;
}
