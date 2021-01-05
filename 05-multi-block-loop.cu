#include <stdio.h>

__global__ void loop()
{
  printf("This is iteration number %d\n", threadIdx.x + (blockIdx.x*blockDim.x));
}

int main()
{
  /*
   * When refactoring `loop` to launch as a kernel, be sure
   * to use the execution configuration to control how many
   * "iterations" to perform.
   *
   * For this exercise, be sure to use more than 1 block in
   * the execution configuration.
   */
  int NUM_BLOCKS = 2;
  int NUM_THREADS = 5;
  loop<<<NUM_BLOCKS, NUM_THREADS>>>();
  cudaDeviceSynchronize();
}

