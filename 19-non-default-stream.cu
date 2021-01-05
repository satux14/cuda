#include <stdio.h>

__global__ void printNumber(int number)
{
  printf("%d\n", number);
}

int main()
{
  for (int i = 0; i < 5; ++i)
  {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    printNumber<<<1, 1, 0, stream>>>(i);
    cudaStreamDestroy(stream);
  }
  cudaDeviceSynchronize();
  
#if 0
  cudaStream_t stream[5];
  
  for (int i = 0; i < 5; ++i)
  {
    cudaStreamCreate(&stream[i]);
  }
  
  for (int i = 0; i < 5; ++i)
  {
    printNumber<<<1, 1, 0, stream[i]>>>(i);
  }

  for (int i = 0; i < 5; ++i)
  {
    cudaStreamDestroy(stream[i]);
  }
  cudaDeviceSynchronize();
#endif
}

