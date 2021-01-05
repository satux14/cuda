#include <stdio.h>

#define N 2048 * 2048 // Number of elements in each vector
//https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_1

__global__ void saxpy(int *a, int *b, int *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < N; i += stride) {
        c[i] = 2 * a[i] + b[i];
    }
}

int main()
{
    int *a, *b, *c;
    int device_id;
    int num_sms;
    
    int size = N * sizeof (int); // The total number of bytes per vector
    
    cudaGetDevice(&device_id);
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device_id);
    
    cudaMallocManaged(&a, size);
    cudaMallocManaged(&b, size);
    cudaMallocManaged(&c, size);

    // Initialize memory
    for( int i = 0; i < N; ++i )
    {
        a[i] = 2;
        b[i] = 1;
        c[i] = 0;
    }

    cudaMemPrefetchAsync(a, size, device_id);
    cudaMemPrefetchAsync(b, size, device_id);
    cudaMemPrefetchAsync(c, size, device_id);

    int threads_per_block = 256;
    int number_of_blocks = num_sms * 32;
    printf("SMs: %d, blocks: %d, threads per block: %d\n", num_sms, number_of_blocks, threads_per_block);
   
    saxpy <<< number_of_blocks, threads_per_block >>> ( a, b, c );
    cudaDeviceSynchronize();

    // Print out the first and last 5 values of c for a quality check
    for( int i = 0; i < 5; ++i )
        printf("c[%d] = %d, ", i, c[i]);
    printf ("\n");
    for( int i = N-5; i < N; ++i )
        printf("c[%d] = %d, ", i, c[i]);
    printf ("\n");

    cudaFree( a ); cudaFree( b ); cudaFree( c );
}

