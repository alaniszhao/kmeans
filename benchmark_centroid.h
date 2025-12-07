#include <stddef.h>

extern unsigned long long core_kernel_cycles;

void benchmark_centroid(size_t n, int k, void **objs, int *clusters, void **centers);
