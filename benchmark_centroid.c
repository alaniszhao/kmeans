#include "benchmark_centroid.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>

#define TILE_SIZE 1024
#define UNROLL 32

unsigned long long core_kernel_cycles = 0;

static __inline__ unsigned long long rdtsc_kernel(void) {
    unsigned hi, lo;
    __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
    return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}

// horizontal sum
static inline float hsum_avx(__m256 v) {
    __m128 v_low = _mm256_castps256_ps128(v);
    __m128 v_high = _mm256_extractf128_ps(v, 1);
    v_low = _mm_add_ps(v_low, v_high);
    v_low = _mm_hadd_ps(v_low, v_low);
    v_low = _mm_hadd_ps(v_low, v_low);
    return _mm_cvtss_f32(v_low);
}

void benchmark_centroid(size_t n, int k, void **objs, int *clusters, void **centers) {
    
    // accumulators
    // k clusters * 3 dimensions * 8 floats
    float* global_acc = _mm_malloc(k * 3 * 8 * sizeof(float), 32);
    memset(global_acc, 0, k * 3 * 8 * sizeof(float));

    int* global_counts = calloc(k, sizeof(int));

    // tile buffers fit in L1 cache
    size_t buffer_size = TILE_SIZE + (k * UNROLL);            // padding (K * 32)
    float* tile_x = _mm_malloc(buffer_size * sizeof(float), 32);
    float* tile_y = _mm_malloc(buffer_size * sizeof(float), 32);
    float* tile_z = _mm_malloc(buffer_size * sizeof(float), 32);
    
    int* tile_counts = malloc(k * sizeof(int));
    int* tile_offsets = malloc(k * sizeof(int));
    int* write_pos = malloc(k * sizeof(int));

    __m256 one = _mm256_set1_ps(1.0f);

    // iterate over tiles
    for (size_t t = 0; t < n; t += TILE_SIZE) {
        size_t current_tile_size = (t + TILE_SIZE > n) ? (n - t) : TILE_SIZE;
        
        memset(tile_counts, 0, k * sizeof(int));
        for (size_t i = 0; i < current_tile_size; i++) {
            tile_counts[clusters[t + i]]++;
        }

        tile_offsets[0] = 0;
        // calculate starting point for each cluster
        for (int c = 1; c < k; c++) {
            int prev_count = tile_counts[c-1];
            // round to 32 so we don't overwrite other clusters' data
            int prev_occupied = (prev_count + 31) & ~31;
            tile_offsets[c] = tile_offsets[c-1] + prev_occupied;
        }

        memcpy(write_pos, tile_offsets, k * sizeof(int));

        // update memory layout to be structure of arrays
        // so we can interleave clusters 
        for (size_t i = 0; i < current_tile_size; i++) {
            int cluster_id = clusters[t + i];
            int pos = write_pos[cluster_id]++;
            
            float* p = (float*)objs[t + i];
            tile_x[pos] = p[0];
            tile_y[pos] = p[1];
            tile_z[pos] = p[2];
        }

        // 0 padding if cluster has <32 points
        for(int c = 0; c < k; c++) {
            int start = tile_offsets[c];
            int count = tile_counts[c];
            int aligned_end = (count + 31) & ~31;
            for(int p = start + count; p < start + aligned_end; p++) {
                tile_x[p] = 0.0f; 
                tile_y[p] = 0.0f; 
                tile_z[p] = 0.0f;
            }
            // update global counts here
            global_counts[c] += count;
        }

        unsigned long long t0 = rdtsc_kernel();

        // interleaved pairs of clusters to fill pipeline
        int c = 0;
        for (; c < k - 1; c += 2) {
            int c1 = c; 
            int c2 = c + 1;
            
            int count1 = tile_counts[c1];
            int count2 = tile_counts[c2];
            if (count1 == 0 && count2 == 0) continue;

            int start1 = tile_offsets[c1];
            int start2 = tile_offsets[c2];
            
            // loop limits rounded up to 16
            int end1 = (count1 + 15) & ~15;
            int end2 = (count2 + 15) & ~15;
            int common_end = (end1 < end2) ? end1 : end2;   // when to stop looping between c1 and c2

            __m256 sx1_0 = _mm256_setzero_ps(); 
            __m256 sx1_1 = _mm256_setzero_ps();
            __m256 sy1_0 = _mm256_setzero_ps(); 
            __m256 sy1_1 = _mm256_setzero_ps();
            __m256 sz1_0 = _mm256_setzero_ps(); 
            __m256 sz1_1 = _mm256_setzero_ps();

            __m256 sx2_0 = _mm256_setzero_ps(); 
            __m256 sx2_1 = _mm256_setzero_ps();
            __m256 sy2_0 = _mm256_setzero_ps();
            __m256 sy2_1 = _mm256_setzero_ps();
            __m256 sz2_0 = _mm256_setzero_ps(); 
            __m256 sz2_1 = _mm256_setzero_ps();

            // interleaved cluster 1 and 2
            int j = 0;
            for (; j < common_end; j += 16) {
                // C1
                sx1_0 = _mm256_fmadd_ps(_mm256_loadu_ps(tile_x + start1 + j),     _mm256_set1_ps(1.0f), sx1_0);
                sx1_1 = _mm256_fmadd_ps(_mm256_loadu_ps(tile_x + start1 + j + 8), _mm256_set1_ps(1.0f), sx1_1);
                sy1_0 = _mm256_fmadd_ps(_mm256_loadu_ps(tile_y + start1 + j),     _mm256_set1_ps(1.0f), sy1_0);
                sy1_1 = _mm256_fmadd_ps(_mm256_loadu_ps(tile_y + start1 + j + 8), _mm256_set1_ps(1.0f), sy1_1);
                sz1_0 = _mm256_fmadd_ps(_mm256_loadu_ps(tile_z + start1 + j),     _mm256_set1_ps(1.0f), sz1_0);
                sz1_1 = _mm256_fmadd_ps(_mm256_loadu_ps(tile_z + start1 + j + 8), _mm256_set1_ps(1.0f), sz1_1);
                // C2
                sx2_0 = _mm256_fmadd_ps(_mm256_loadu_ps(tile_x + start2 + j),     _mm256_set1_ps(1.0f), sx2_0);
                sx2_1 = _mm256_fmadd_ps(_mm256_loadu_ps(tile_x + start2 + j + 8), _mm256_set1_ps(1.0f), sx2_1);
                sy2_0 = _mm256_fmadd_ps(_mm256_loadu_ps(tile_y + start2 + j),     _mm256_set1_ps(1.0f), sy2_0);
                sy2_1 = _mm256_fmadd_ps(_mm256_loadu_ps(tile_y + start2 + j + 8), _mm256_set1_ps(1.0f), sy2_1);
                sz2_0 = _mm256_fmadd_ps(_mm256_loadu_ps(tile_z + start2 + j),     _mm256_set1_ps(1.0f), sz2_0);
                sz2_1 = _mm256_fmadd_ps(_mm256_loadu_ps(tile_z + start2 + j + 8), _mm256_set1_ps(1.0f), sz2_1);
            }

            // end of cluster 1
            for (; j < end1; j += 16) {
                sx1_0 = _mm256_add_ps(sx1_0, _mm256_loadu_ps(tile_x + start1 + j));
                sx1_1 = _mm256_add_ps(sx1_1, _mm256_loadu_ps(tile_x + start1 + j + 8));
                sy1_0 = _mm256_add_ps(sy1_0, _mm256_loadu_ps(tile_y + start1 + j));
                sy1_1 = _mm256_add_ps(sy1_1, _mm256_loadu_ps(tile_y + start1 + j + 8));
                sz1_0 = _mm256_add_ps(sz1_0, _mm256_loadu_ps(tile_z + start1 + j));
                sz1_1 = _mm256_add_ps(sz1_1, _mm256_loadu_ps(tile_z + start1 + j + 8));
            }
            // end of cluster 2
            for (; j < end2; j += 16) {
                sx2_0 = _mm256_add_ps(sx2_0, _mm256_loadu_ps(tile_x + start2 + j));
                sx2_1 = _mm256_add_ps(sx2_1, _mm256_loadu_ps(tile_x + start2 + j + 8));
                sy2_0 = _mm256_add_ps(sy2_0, _mm256_loadu_ps(tile_y + start2 + j));
                sy2_1 = _mm256_add_ps(sy2_1, _mm256_loadu_ps(tile_y + start2 + j + 8));
                sz2_0 = _mm256_add_ps(sz2_0, _mm256_loadu_ps(tile_z + start2 + j));
                sz2_1 = _mm256_add_ps(sz2_1, _mm256_loadu_ps(tile_z + start2 + j + 8));
            }

            // accumulate to global
            float* g1 = &global_acc[c1 * 24]; // 3 dims * 8 floats
            _mm256_store_ps(g1,     _mm256_add_ps(_mm256_load_ps(g1),    _mm256_add_ps(sx1_0, sx1_1)));
            _mm256_store_ps(g1+8,   _mm256_add_ps(_mm256_load_ps(g1+8),  _mm256_add_ps(sy1_0, sy1_1)));
            _mm256_store_ps(g1+16,  _mm256_add_ps(_mm256_load_ps(g1+16), _mm256_add_ps(sz1_0, sz1_1)));

            float* g2 = &global_acc[c2 * 24];
            _mm256_store_ps(g2,     _mm256_add_ps(_mm256_load_ps(g2),    _mm256_add_ps(sx2_0, sx2_1)));
            _mm256_store_ps(g2+8,   _mm256_add_ps(_mm256_load_ps(g2+8),  _mm256_add_ps(sy2_0, sy2_1)));
            _mm256_store_ps(g2+16,  _mm256_add_ps(_mm256_load_ps(g2+16), _mm256_add_ps(sz2_0, sz2_1)));
        }

        // do last cluster if k is odd
        if (c < k) {
            int start = tile_offsets[c];
            int count = tile_counts[c];
            int end = (count + 15) & ~15;
            
            __m256 sx = _mm256_setzero_ps(); 
            __m256 sy = _mm256_setzero_ps(); 
            __m256 sz = _mm256_setzero_ps();
            
            for(int j =0 ; j < end; j += 16) {
                sx = _mm256_add_ps(sx, _mm256_add_ps(_mm256_loadu_ps(tile_x+start+j), _mm256_loadu_ps(tile_x+start+j+8)));
                sy = _mm256_add_ps(sy, _mm256_add_ps(_mm256_loadu_ps(tile_y+start+j), _mm256_loadu_ps(tile_y+start+j+8)));
                sz = _mm256_add_ps(sz, _mm256_add_ps(_mm256_loadu_ps(tile_z+start+j), _mm256_loadu_ps(tile_z+start+j+8)));
            }
            float* g = &global_acc[c * 24];
            _mm256_store_ps(g,    _mm256_add_ps(_mm256_load_ps(g),    sx));
            _mm256_store_ps(g+8,  _mm256_add_ps(_mm256_load_ps(g+8),  sy));
            _mm256_store_ps(g+16, _mm256_add_ps(_mm256_load_ps(g+16), sz));
        }

        unsigned long long t1 = rdtsc_kernel();
        core_kernel_cycles += (t1 - t0);
    }


    // horizontal sum of all lanes for each dimension 
    // divide by count
    for (int i = 0; i < k; i++) {
        float* acc_base = &global_acc[i * 24];
        float sum_x = hsum_avx(_mm256_load_ps(acc_base));
        float sum_y = hsum_avx(_mm256_load_ps(acc_base + 8));
        float sum_z = hsum_avx(_mm256_load_ps(acc_base + 16));

        int count = global_counts[i];
        float* center = (float*)centers[i];

        if (count > 0) {
            float scale = 1.0f / count;
            center[0] = sum_x * scale;
            center[1] = sum_y * scale;
            center[2] = sum_z * scale;
        } 
    }

    // cleanup
    _mm_free(global_acc);
    _mm_free(tile_x);
    _mm_free(tile_y);
    _mm_free(tile_z);
    free(global_counts);
    free(tile_counts);
    free(tile_offsets);
    free(write_pos);
}