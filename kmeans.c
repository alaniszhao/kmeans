/*-------------------------------------------------------------------------
*
* kmeans.c
*    Generic k-means implementation
*
* Copyright (c) 2016, Paul Ramsey <pramsey@cleverelephant.ca>
*
*------------------------------------------------------------------------*/

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "distance_kernel.h"
#include "kmeans.h"

#define BATCH 56

#define BATCH 56  /* kernel batch size */

/* New update_r using distance_kernel batching */
static void update_r(kmeans_config *config)
{
    assert(config->objs);
    assert(config->centers);
    assert(config->clusters);

    float data_x[BATCH], data_y[BATCH], data_z[BATCH];
    float dist_out[BATCH];
    float best_distances[config->num_objs];

    for (size_t i = 0; i < config->num_objs; i++)
        best_distances[i] = FLT_MAX;

    int num_batches = (config->num_objs + BATCH - 1) / BATCH;

    for (int b = 0; b < num_batches; b++)
    {
        int batch_start = b * BATCH;
        int batch_end = batch_start + BATCH;
        if (batch_end > config->num_objs)
            batch_end = config->num_objs;

        int batch_size = batch_end - batch_start;

        /* Load current batch into SoA format */
        for (int i = 0; i < batch_size; i++)
        {
            int idx = batch_start + i;
            float *p = (float*)config->objs[idx];
            if (p)
            {
                data_x[i] = p[0];
                data_y[i] = p[1];
                data_z[i] = p[2];
            }
            else
            {
                data_x[i] = data_y[i] = data_z[i] = 0.0f;
            }
        }
        /* Pad remaining entries */
        for (int i = batch_size; i < BATCH; i++)
            data_x[i] = data_y[i] = data_z[i] = 0.0f;

        /* Compare against each centroid using distance_kernel */
        for (int c = 0; c < config->k; c++)
        {
            float *cent = (float*)config->centers[c];

            distance_kernel(dist_out,
                            data_x, data_y, data_z,
                            cent[0], cent[1], cent[2]);

            /* Update nearest cluster */
            for (int i = 0; i < batch_size; i++)
            {
                int idx = batch_start + i;
                if (!config->objs[idx])
                {
                    config->clusters[idx] = KMEANS_NULL_CLUSTER;
                    continue;
                }

                float d = dist_out[i];
                if (d < best_distances[idx])
                {
                    best_distances[idx] = d;
                    config->clusters[idx] = c;
                }
            }
        }
    }
}

static void
update_means(kmeans_config *config)
{
	int i;

	for (i = 0; i < config->k; i++)
	{
		/* Update the centroid for this cluster */
		(config->centroid_method)(config->objs, config->clusters, config->num_objs, i, config->centers[i]);
	}
}

kmeans_result
kmeans(kmeans_config *config)
{
	int iterations = 0;
	int *clusters_last;
	size_t clusters_sz = sizeof(int)*config->num_objs;

	assert(config);
	assert(config->objs);
	assert(config->num_objs);
	assert(config->centroid_method);
	assert(config->centers);
	assert(config->k);
	assert(config->clusters);
	assert(config->k <= config->num_objs);

	/* Zero out cluster numbers, just in case user forgets */
	memset(config->clusters, 0, clusters_sz);

	/* Set default max iterations if necessary */
	if (!config->max_iterations)
		config->max_iterations = KMEANS_MAX_ITERATIONS;

	/*
	 * Previous cluster state array. At this time, r doesn't mean anything
	 * but it's ok
	 */
	clusters_last = kmeans_malloc(clusters_sz);

	while (1)
	{
		/* Store the previous state of the clustering */
		memcpy(clusters_last, config->clusters, clusters_sz);
		update_r(config);
		update_means(config);
		/*
		 * if all the cluster numbers are unchanged since last time,
		 * we are at a stable solution, so we can stop here
		 */
		if (memcmp(clusters_last, config->clusters, clusters_sz) == 0)
		{
			kmeans_free(clusters_last);
			config->total_iterations = iterations;
			return KMEANS_OK;
		}

		if (iterations++ > config->max_iterations)
		{
			kmeans_free(clusters_last);
			config->total_iterations = iterations;
			return KMEANS_EXCEEDED_MAX_ITERATIONS;
		}
	}

	kmeans_free(clusters_last);
	config->total_iterations = iterations;
	return KMEANS_ERROR;
}


