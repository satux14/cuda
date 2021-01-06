#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"
#include "files.h"

#define SOFTENING 1e-9f

/*
 * Each body contains x, y, and z coordinate positions,
 * as well as velocities in the x, y, and z directions.
 */

typedef struct {
    float x, y, z, vx, vy, vz;
} Body;

/*
 * Calculate the gravitational impact of all bodies in the system
 * on all others.
 */

int *max_count;

__global__
void bodyForce(Body *p, float dt, int n, int *max) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  //for (int i = 0; i < n; ++i) {
  if (i < n) {
    float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

    for (int j = 0; j < n; j++) {
      float dx = p[j].x - p[i].x;
      float dy = p[j].y - p[i].y;
      float dz = p[j].z - p[i].z;
      float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
      float invDist = rsqrtf(distSqr);
      float invDist3 = invDist * invDist * invDist;

      Fx += dx * invDist3;
      Fy += dy * invDist3;
      Fz += dz * invDist3;
    }

    p[i].vx += dt*Fx;
    p[i].vy += dt*Fy;
    p[i].vz += dt*Fz;
    (*max)++;
  }
}

__global__
void integrate(Body *p, float dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {

      p[i].x += p[i].vx*dt;
      p[i].y += p[i].vy*dt;
      p[i].z += p[i].vz*dt;
    }
}

int main(const int argc, const char** argv) {

  int deviceId;
  
  cudaGetDevice(&deviceId);
  
  cudaError_t err, sync_err, async_err;
  // The assessment will test against both 2<11 and 2<15.
  // Feel free to pass the command line argument 15 when you gernate ./nbody report files
  int nBodies = 2<<11;
  if (argc > 1) nBodies = 2<<atoi(argv[1]);

  // The assessment will pass hidden initialized values to check for correctness.
  // You should not make changes to these files, or else the assessment will not work.
  const char * initialized_values;
  const char * solution_values;

  if (nBodies == 2<<11) {
    initialized_values = "files/initialized_4096";
    solution_values = "files/solution_4096";
  } else { // nBodies == 2<<15
    initialized_values = "files/initialized_65536";
    solution_values = "files/solution_65536";
  }

  if (argc > 2) initialized_values = argv[2];
  if (argc > 3) solution_values = argv[3];

  const float dt = 0.01f; // Time step
  const int nIters = 10;  // Simulation iterations

  int bytes = nBodies * sizeof(Body);
  float *buf;

  //buf = (float *)malloc(bytes);
  err = cudaMallocManaged(&buf, bytes);
  if (err != cudaSuccess)
  {
      printf("cuda Malloc failed: %s\n", cudaGetErrorString(err));
      return -1;
  }
  cudaMallocManaged(&max_count, sizeof(int));
  *max_count = 0;

  Body *p = (Body*)buf;
  
  cudaMemPrefetchAsync(buf, bytes, cudaCpuDeviceId);
  read_values_from_file(initialized_values, buf, bytes);

  double totalTime = 0.0;

  /*
   * This simulation will run for 10 cycles of time, calculating gravitational
   * interaction amongst bodies, and adjusting their positions to reflect.
   */

  for (int iter = 0; iter < nIters; iter++) {
    StartTimer();

    int threads_per_block = 128;
    int number_of_blocks = (nBodies / threads_per_block);
    cudaMemPrefetchAsync(buf, bytes, deviceId);
    bodyForce <<< number_of_blocks, threads_per_block >>> ( p, dt, nBodies, max_count );
    sync_err = cudaGetLastError();
    async_err = cudaDeviceSynchronize();
 
    if (sync_err != cudaSuccess || async_err != cudaSuccess) {
        printf("Sync error: %s\n", cudaGetErrorString(sync_err));
        printf("Async error: %s\n", cudaGetErrorString(async_err));
    }

  /*
   * This position integration cannot occur until this round of `bodyForce` has completed.
   * Also, the next round of `bodyForce` cannot begin until the integration is complete.
   */
    integrate <<< number_of_blocks, threads_per_block >>> ( p, dt, nBodies );
    sync_err = cudaGetLastError();
    async_err = cudaDeviceSynchronize();
 
    if (sync_err != cudaSuccess || async_err != cudaSuccess) {
        printf("i-Sync error: %s\n", cudaGetErrorString(sync_err));
        printf("i-Async error: %s\n", cudaGetErrorString(async_err));
    }

    const double tElapsed = GetTimer() / 1000.0;
    totalTime += tElapsed;
  }

  cudaMemPrefetchAsync(buf, bytes, cudaCpuDeviceId);
  double avgTime = totalTime / (double)(nIters);
  float billionsOfOpsPerSecond = 1e-9 * nBodies * nBodies / avgTime;
  write_values_to_file(solution_values, buf, bytes);

  // You will likely enjoy watching this value grow as you accelerate the application,
  // but beware that a failure to correctly synchronize the device might result in
  // unrealistically high values.
  printf("%0.3f Billion Interactions / second", billionsOfOpsPerSecond);
  
  printf("\n nBodies: %d, Max Count: %d\n", nBodies, *max_count);

  cudaFree(max_count);
  cudaFree(buf);
}

# Standalone compilation: nvcc -std=c++11 21-nbody.cu
