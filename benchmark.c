#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "testcases/im_2018.h" //testcase
#include "kmeans.h"
#include "kmeans_baseline.h"

#include <math.h>

typedef struct point
{
	float x;
	float y;
    float z;
} point;

#define RUNS 100

static __inline__ unsigned long long rdtsc(void) {
    unsigned hi, lo;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return ((unsigned long long)lo) | (((unsigned long long)hi) << 32);
}

static int verify_results(kmeans_baseline_config *ref,
                          kmeans_config *test)
{
    const float EPS = 1e-3f;
    int i, c;

    for (i = 0; i < ref->num_objs; i++) {
        if (ref->clusters[i] != test->clusters[i]) {
            printf("cluster mismatch\n");
            return 0;
        }
    }

    for (c = 0; c < ref->k; c++) {
        float *cref = (float *)ref->centers[c];
        float *ctst = (float *)test->centers[c];

        for (int d = 0; d < 3; d++) {
            float diff = fabsf(cref[d] - ctst[d]);
            if (diff > EPS) {
                printf("centroid mismatch\n");
                return 0;
            }
        }
    }
    return 1;
}

static double pt_distance(const Pointer a, const Pointer b)
{
	point *pa = (point*)a;
	point *pb = (point*)b;

	float dx = (pa->x - pb->x);
	float dy = (pa->y - pb->y);
    float dz = (pa->z - pb->z);

	return dx*dx + dy*dy + dz*dz;
}

static void pt_centroid(const Pointer * objs, const int * clusters, size_t num_objs, int cluster, Pointer centroid)
{
	int i;
	int num_cluster = 0;
	point sum;
	point **pts = (point**)objs;
	point *center = (point*)centroid;

	sum.x = sum.y = sum.z = 0.0;

	if (num_objs <= 0) return;

	for (i = 0; i < num_objs; i++)
	{
		if (clusters[i] != cluster) continue;

		sum.x += pts[i]->x;
		sum.y += pts[i]->y;
        sum.z += pts[i]->z;
		num_cluster++;
	}
	if (num_cluster)
	{
		sum.x /= num_cluster;
		sum.y /= num_cluster;
        sum.z /= num_cluster;
		*center = sum;
	}
	return;
}

int main(int nargs, char **args)
{
    int k_vals[4] = {3, 5, 7, 10};
    for (int k_i = 0; k_i < 4; k_i++) {
        float scales[4] = {0.25, 0.5, 0.75, 1.0};
        int k = k_vals[k_i];
        printf("%d kmeans baseline\n", k);

        for (int s = 0; s < 4; s++) {
            int H = (int)(IMAGE_HEIGHT * scales[s]);
            int W = (int)(IMAGE_WIDTH  * scales[s]);
            int num_pts = H * W;

            kmeans_config config;
            config.num_objs = num_pts;
            config.k = k;
            config.max_iterations = 100;
            config.distance_method = NULL;
            config.centroid_method = pt_centroid;

            config.objs = calloc(num_pts, sizeof(Pointer));
            config.centers = calloc(k, sizeof(Pointer));
            config.clusters = calloc(num_pts, sizeof(int));

            kmeans_baseline_config config_baseline;
            config_baseline.num_objs = num_pts;
            config_baseline.k = k;
            config_baseline.max_iterations = 100;
            config_baseline.distance_method = pt_distance;
            config_baseline.centroid_method = pt_centroid;

            config_baseline.objs = calloc(num_pts, sizeof(Pointer));
            config_baseline.centers = calloc(k, sizeof(Pointer));
            config_baseline.clusters = calloc(num_pts, sizeof(int));
            
            // load scaled image
            int idx = 0;
            for (int r = 0; r < H; r++) {
                for (int c = 0; c < W; c++) {
                    int full_idx = r*IMAGE_WIDTH + c;
                    point *p = malloc(sizeof(point));
                    p->x = im_2018[full_idx][0];
                    p->y = im_2018[full_idx][1];
                    p->z = im_2018[full_idx][2];

                    config.objs[idx] = p;
                    idx++;
                }
            }

            // init centroids
            for (int i = 0; i < k; i++) {
                float *c = malloc(sizeof(point));
                memcpy(c, config.objs[i], sizeof(point));
                config.centers[i] = c;
            }

            // load scaled baseline image
            idx = 0;
            for (int r = 0; r < H; r++) {
                for (int c = 0; c < W; c++) {
                    int full_idx = r*IMAGE_WIDTH + c;
                    point *pb = malloc(sizeof(point));
                    pb->x = im_2018[full_idx][0];
                    pb->y = im_2018[full_idx][1];
                    pb->z = im_2018[full_idx][2];

                    config_baseline.objs[idx] = pb;
                    idx++;
                }
            }

            // load baseline centroids
            for (int i = 0; i < k; i++) {
                point *cb = malloc(sizeof(point));
                memcpy(cb, config_baseline.objs[i], sizeof(point));
                config_baseline.centers[i] = cb;
            }

            // verify implementation
            kmeans_result result = kmeans(&config);
            kmeans_baseline_result result_baseline = kmeans_baseline(&config_baseline);
            verify_results(&config_baseline, &config);

            // time implementation vs baseline
            unsigned long long kmeans_total = 0;
            unsigned long long base_total = 0;
            unsigned long long kmeanst_0, kmeanst_1, baset_0, baset_1;
            memset(config.clusters, 0, num_pts * sizeof(int));
            for (int i = 0; i < RUNS; i++) {
                kmeanst_0 = rdtsc();
                result = kmeans(&config);
                kmeanst_1 = rdtsc();
                baset_0 = rdtsc();
                result_baseline = kmeans_baseline(&config_baseline);
                baset_1 = rdtsc();
                kmeans_total = kmeans_total + (kmeanst_1 - kmeanst_0);
                base_total = base_total + (baset_1 - baset_0);
            }
            unsigned long long kmeans_avg = kmeans_total/RUNS;
            unsigned long long base_avg   = base_total/RUNS;
            printf("%llu %llu\n", kmeans_avg, base_avg);

            // cleanup
            for (int i = 0; i < num_pts; i++) free(config.objs[i]);
            for (int i = 0; i < k; i++) free(config.centers[i]);
            free(config.objs);
            free(config.centers);
            free(config.clusters);
            for (int i = 0; i < num_pts; i++) free(config_baseline.objs[i]);
            for (int i = 0; i < k; i++) free(config_baseline.centers[i]);
            free(config_baseline.objs);
            free(config_baseline.centers);
            free(config_baseline.clusters);
        }
    }
    return 0;
}