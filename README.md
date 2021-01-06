# CUDA Programming
 - A collection of thread is a block
 - A collection of blocks associated with a given kernel launch is a
   grid
 - GPU functions are called kernels
 - Kernels are launched with execution configuration
   (gpu_function<<1,2>>()) => <<<number of blocks in a grid, number of
   thread in each block>>>

*someKernel<<<1, 1>>() is configured to run in a single thread block which has a single thread and will therefore run only once.</br>
someKernel<<<1, 10>>() is configured to run in a single thread block which has 10 threads and will therefore run 10 times.</br>
someKernel<<<10, 1>>() is configured to run in 10 thread blocks which each have a single thread and will therefore run 10 times.</br>
someKernel<<<10, 10>>() is configured to run in 10 thread blocks which each have 10 threads and will therefore run 100 times.*</br>

## Variables:

 - gridDim.x - Number of blocks in the grid
 - blockIdx.x - Index of current block within the grid
 - blockDim.x - Number of threads in a block
 - threadIdx.x - Index of the thread within a block
 - CUDA kernels are executed in a grid of 1 or more blocks, with each
   block containing the same number of 1 or more threads.
 - There is a limit to the number of threads that can exist in a thread
   block: 1024 to be precise.
 - threadIdx.x + blockIdx.x * blockDim.x => Iterate through each data
   element of a vector.

## Memory:
https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#memory-optimizations</br>
malloc(size) => cudaMallocManaged(&a, size)
free(a) => cudaFree(a)

## Block config mismatch to number of needed threads:

 - Maximum number of threads : 256

    int N = 100000;
    size_t threads_per_block = 256;
    // Ensure there are at least `N` threads in the grid, but only 1 block's worth extra
    size_t number_of_blocks = (N + threads_per_block - 1) / threads_per_block;
    some_kernel<<<number_of_blocks, threads_per_block>>>(N);

    __global__ some_kernel(int N)
    {
      int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) // Check to make sure `idx` maps to some value within `N`
      {
        // Only do work if it does
      }
    }

## Grid Stride when data set is greater than Grid:

    __global void kernel(int *a, int N)
    {
      int indexWithinTheGrid = threadIdx.x + blockIdx.x * blockDim.x;
      int gridStride = gridDim.x * blockDim.x;
    for (int i = indexWithinTheGrid; i < N; i += gridStride)
      {
        // do work on a[i];
      }
    }

## Error handling:

 - Most CUDA function return cudaError_t and get string of the error
   using cudaGetErrorString()
 - For kernel launch errors: use cudaGetLastError() to get last error
   value and use cudaGetErrorString()

### Sample CUDA error wrapper:

    inline cudaError_t checkCuda(cudaError_t result)
    {
      if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
      }
      return result;
    }

## CUDA Streams:

> cudaStream_t stream; cudaStreamCreate(&stream); 
> someKernel<<<number_of_blocks, threads_per_block, 0, stream>>>(); 
> cudaStreamDestroy(stream);

    Ex:
      cudaStream_t stream1, stream2, stream3;
      cudaStreamCreate(&stream1);
      initWith<<<numberOfBlocks, threadsPerBlock, 0, stream1>>>(3, a, N);
      cudaStreamCreate(&stream2);
      initWith<<<numberOfBlocks, threadsPerBlock, 0, stream2>>>(4, b, N);
      cudaStreamCreate(&stream3);
      initWith<<<numberOfBlocks, threadsPerBlock, 0, stream3>>>(0, c, N);


## Mem Prefetch:

 - cudaMemPrefetchAsync(a, size, deviceId); => Before GPU accessing
   memory
 - cudaMemPrefetchAsync(c, size, cudaCpuDeviceId); => Before CPU
   accessing memory

## Manual memory handling:

 - cudaMalloc will allocate memory directly to the active GPU.

> 	This prevents all GPU page faults. In exchange, the pointer it
> returns is not available for access by host code.

 - cudaMallocHost will allocate memory directly to the CPU.

> 	It also "pins" the memory, or page locks it, which will allow for
> asynchronous copying of the memory to and from a GPU. Too much pinned
> memory can interfere with CPU performance, so use it only with
> intention. Pinned memory should be freed with cudaFreeHost. cudaMemcpy
> can copy (not transfer) memory, either from host to device or from
> device to host.

    int *host_a, *device_a;        // Define host-specific and device-specific arrays.
    cudaMalloc(&device_a, size);   // `device_a` is immediately available on the GPU.
    cudaMallocHost(&host_a, size); // `host_a` is immediately available on CPU, and is page-locked, or pinned.
    
    initializeOnHost(host_a, N);   // No CPU page faulting since memory is already allocated on the host.
    
    // `cudaMemcpy` takes the destination, source, size, and a CUDA-provided variable for the direction of the copy.
    cudaMemcpy(device_a, host_a, size, cudaMemcpyHostToDevice);
    
    kernel<<<blocks, threads, 0, someStream>>>(device_a, N);
    
    // `cudaMemcpy` can also copy data from device to host.
    cudaMemcpy(host_a, device_a, size, cudaMemcpyDeviceToHost);
    
    verifyOnHost(host_a, N);
    
    cudaFree(device_a);
    cudaFreeHost(host_a);          // Free pinned memory like this.

 - Use cudaMemcpyAsync for async copy if host memory is pinned
   (allocated by cudaMallocHost)
    int N = 2<<24;
    int size = N * sizeof(int);
    
    int *host_array;
    int *device_array;
    
    cudaMallocHost(&host_array, size);               // Pinned host memory allocation.
    cudaMalloc(&device_array, size);                 // Allocation directly on the active GPU device.
    
    initializeData(host_array, N);                   // Assume this application needs to initialize on the host.
    
    const int numberOfSegments = 4;                  // This example demonstrates slicing the work into 4 segments.
    int segmentN = N / numberOfSegments;             // A value for a segment's worth of `N` is needed.
    size_t segmentSize = size / numberOfSegments;    // A value for a segment's worth of `size` is needed.
    
    // For each of the 4 segments...
    for (int i = 0; i < numberOfSegments; ++i)
    {
      // Calculate the index where this particular segment should operate within the larger arrays.
      segmentOffset = i * segmentN;
    
      // Create a stream for this segment's worth of copy and work.
      cudaStream_t stream;
      cudaStreamCreate(&stream);
      
      // Asynchronously copy segment's worth of pinned host memory to device over non-default stream.
      cudaMemcpyAsync(&device_array[segmentOffset],  // Take care to access correct location in array.
                      &host_array[segmentOffset],    // Take care to access correct location in array.
                      segmentSize,                   // Only copy a segment's worth of memory.
                      cudaMemcpyHostToDevice,
                      stream);                       // Provide optional argument for non-default stream.
                      
      // Execute segment's worth of work over same non-default stream as memory copy.
      kernel<<<number_of_blocks, threads_per_block, 0, stream>>>(&device_array[segmentOffset], segmentN);
      
      // `cudaStreamDestroy` will return immediately (is non-blocking), but will not actually destroy stream until
      // all stream operations are complete.
      cudaStreamDestroy(stream);
    }

## Advanced:
https://developer.download.nvidia.com/CUDA/training/StreamsAndConcurrencyWebinar.pdf

    cudaStream_t stream1, stream2, stream3, stream4 ;
    cudaStreamCreate ( &stream1) ;
    ...
    cudaMalloc ( &dev1, size ) ;
    cudaMallocHost ( &host1, size ) ; // pinned memory required on host
    …
    cudaMemcpyAsync ( dev1, host1, size, H2D, stream1 ) ;
    kernel2 <<< grid, block, 0, stream2 >>> ( …, dev2, … ) ;
    kernel3 <<< grid, block, 0, stream3 >>> ( …, dev3, … ) ;
    cudaMemcpyAsync ( host4, dev4, size, D2H, stream4 ) ;
    some_CPU_method ();
    ...
    Asynchronous:
    Note: Data used by concurrent operations should be independent

