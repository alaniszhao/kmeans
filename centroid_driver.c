#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>
#include "benchmark_centroid.h"

void** generate_data(int n) {
    void** objs = malloc(n * sizeof(void*));
    for (int i = 0; i < n; i++) {
        float* p = malloc(3 * sizeof(float));
        p[0] = (float)rand() / RAND_MAX;
        p[1] = (float)rand() / RAND_MAX;
        p[2] = (float)rand() / RAND_MAX;
        objs[i] = p;
    }
    return objs;
}

void free_data(void** objs, int n) {
    for (int i = 0; i < n; i++) free(objs[i]);
    free(objs);
}

int main() {
    int n_values[] = {56, 560, 1120, 2240, 4480, 8960, 17920, 179200};
    int num_n_tests = sizeof(n_values) / sizeof(n_values[0]);

    int k_values[] = {3, 5, 7, 10, 20};
    int num_k_tests = sizeof(k_values) / sizeof(k_values[0]);

    printf("N,K,TotalCycles,CoreCycles,CoreFLOPS_Per_Cycle\n");

    for (int i = 0; i < num_n_tests; i++) {
        int n = n_values[i];
        void** objs = generate_data(n);

        for (int j = 0; j < num_k_tests; j++) {
            int k = k_values[j];

            int* clusters = malloc(n * sizeof(int));
            void** centers = malloc(k * sizeof(void*));
            for(int c=0; c<k; c++) centers[c] = malloc(3 * sizeof(float));

            for (int p = 0; p < n; p++) clusters[p] = rand() % k;

            benchmark_centroid(n, k, objs, clusters, centers);

            int iterations = 1000;
            if (n > 50000) iterations = 100; 

            core_kernel_cycles = 0;

            for (int iter = 0; iter < iterations; iter++) {
                benchmark_centroid(n, k, objs, clusters, centers);
            }

            double avg_core_cycles = (double)core_kernel_cycles / iterations;
            double total_flops = n * 6.0;
            double core_flops_per_cycle = total_flops / avg_core_cycles;

            printf("%d,%d,-,%.0f,%.2f\n", n, k, avg_core_cycles, core_flops_per_cycle);

            free(clusters);
            for(int c=0; c<k; c++) free(centers[c]);
            free(centers);
        }
        free_data(objs, n);
    }
    return 0;
}