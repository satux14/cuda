#include <stdio.h>

__global__
void deviceKernel(int *a, int N)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = idx; i < N; i += stride)
  {
    a[i] = 1;
  }
}

void hostFunction(int *a, int N)
{
  for (int i = 0; i < N; ++i)
  {
    a[i] = 1;
  }
}

int main()
{

  int N = 2<<24;
  size_t size = N * sizeof(int);
  int *a;
  cudaMallocManaged(&a, size);

  /*
   * Conduct experiments to learn more about the behavior of
   * `cudaMallocManaged`.
   *
   * What happens when unified memory is accessed only by the GPU?
   * What happens when unified memory is accessed only by the CPU?
   * What happens when unified memory is accessed first by the GPU then the CPU?
   * What happens when unified memory is accessed first by the CPU then the GPU?
   *
   * Hypothesize about UM behavior, page faulting specificially, before each
   * experiment, and then verify by running `nsys`.
   */
#define GPU
#define CPU


#ifdef CPU
  hostFunction(a, N);
  printf("CPU Execution done\n");
#endif

#ifdef GPU
  int device_id;
  cudaDeviceProp props;
  cudaGetDevice(&device_id);
  cudaGetDeviceProperties(&props, device_id);
  size_t number_of_blocks = (N + props.maxThreadsPerBlock - 1)/props.maxThreadsPerBlock;

   deviceKernel<<<number_of_blocks, props.maxThreadsPerBlock>>>(a, N);
   cudaDeviceSynchronize();
   printf("GPU Execution done\n");
#endif


  cudaFree(a);
}

