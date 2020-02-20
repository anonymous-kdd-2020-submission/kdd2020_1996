CC = gcc
#Using -Ofast instead of -O3 might result in faster code, but is supported only by newer GCC versions
CFLAGS = -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result

all: sphere_tree

sphere_tree : sphere_tree.c
	$(CC) sphere_tree.c -o sphere_tree $(CFLAGS)

clean:
	rm -rf sphere_tree
	