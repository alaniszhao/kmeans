CC = gcc
CFLAGS = -mavx2 -mfma -O3 -std=c99 -ffast-math -g
LDLIBS = -lm

OBJS = kmeans.o kmeans_baseline.o distance_kernel.o
EXE = benchmark

all: $(EXE)

run: $(EXE)
	./benchmark

clean:
	@rm -f *.o $(EXE)

benchmark: $(OBJS) benchmark.o
	$(CC) $(CFLAGS) $^ $(LDLIBS) -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@