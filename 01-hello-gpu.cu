#include <stdio.h>

void helloCPU()
{
  printf("Hello from the CPU.\n");
}

/* CUDA function : Device Code - __global__ keyword */
__global__ void helloGPU()
{
  printf("Hello also from the GPU.\n");
}

int main()
{
  helloCPU();
  
  /* Call the GPU function */
  helloGPU<<<1, 1>>>();
  
  helloCPU();
  cudaDeviceSynchronize();
}

/* nvcc -arch=sm_70 -o hello-gpu 01-hello/01-hello-gpu.cu -run */

