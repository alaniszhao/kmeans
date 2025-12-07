run this implementation by doing ssh-ing into ece006.ece.local.cmu.edu and doing make run. 

distance_driver.c:
benchmarks the distance kernel for a variety of input sizes
you can run this with 
gcc -mavx2 -mfma -O3 -std=c99 -ffast-math -o distance_driver.x distance_driver.c distance_kernel.c
./distance_driver.x

distance_kernel.c/h:
implementation of distance kernel

centroid_driver.c:
benchmarks the centroid kernel for a variety of input sizes
gcc -mavx2 -mfma -O3 -std=c99 -ffast-math centroid_driver.c benchmark_centroid.c -lm -o centroid_driver
./centroid_driver 

centroid_kernel.c/h:
implementation of centroid kernel

benchmark.c:
benchmarks our kmeans implementation against the baseline

kmeans_baseline.c/h:
baseline implementation of kmeans

testcases:
contains jpg/mat/c arr format of 5 images from our test dataset. also contains 
clusters.py file which prints out the # of human-annotated clusters