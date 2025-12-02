#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "testcases/im_2018.h" //testcase
#include "kmeans.h"
#include "kmeans_baseline.h"

#include <math.h>

static int verify_results(kmeans_baseline_config *ref, kmeans_config *test)
{
    const float EPS = 1e-3f;
    int i, c;

    for (i = 0; i < ref->num_objs; i++)
    {
        if (ref->clusters[i] != test->clusters[i])
        {
            printf("Cluster mismatch at index %d: ref=%d test=%d\n",
                   i, ref->clusters[i], test->clusters[i]);
            return 0;
        }
    }

    for (c = 0; c < ref->k; c++)
    {
        float *cref = (float*)ref->centers[c];
        float *ctst = (float*)test->centers[c];

        for (int d = 0; d < 3; d++)
        {
            float diff = fabsf(cref[d] - ctst[d]);
            if (diff > EPS)
            {
                printf("Centroid mismatch: c=%d dim=%d diff=%f\n", c, d, diff);
                return 0;
            }
        }
    }

    printf("Results match!\n");
    return 1;
}

typedef struct point
{
	float x;
	float y;
    float z;
} point;


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
    int total_pixels = IMAGE_HEIGHT * IMAGE_WIDTH;
    int num_pts = total_pixels;
    int k = 3;

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

    // load image
    for (int i = 0; i < num_pts; i++) {
        float *p = malloc(3 * sizeof(float));
        p[0] = (float)im_2018[i][0];
        p[1] = (float)im_2018[i][1];
        p[2] = (float)im_2018[i][2];
        config.objs[i] = p;
    }

    // init centroids
    for (int i = 0; i < k; i++) {
        float *c = malloc(3 * sizeof(float));
        memcpy(c, config.objs[i], 3 * sizeof(float));
        config.centers[i] = c;
    }

    // load baseline image
    for (int i = 0; i < num_pts; i++) {
        point *pb = malloc(sizeof(point));
        pb->x = im_2018[i][0];
        pb->y = im_2018[i][1];
        pb->z = im_2018[i][2];
        config_baseline.objs[i] = pb;
    }

    // load baseline centroids
    for (int i = 0; i < k; i++) {
        point *cb = malloc(sizeof(point));
        memcpy(cb, config_baseline.objs[i], sizeof(point));
        config_baseline.centers[i] = cb;
    }

    kmeans_result result = kmeans(&config);
    kmeans_baseline_result result_baseline = kmeans_baseline(&config_baseline);
    verify_results(&config_baseline, &config);

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

    return 0;
}