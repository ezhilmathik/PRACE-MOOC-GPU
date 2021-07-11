## Week 3: CUDA (advanced): Numerical Algebra, Advanced Topics, Profiling and Tuning

This week, we discuss advanced topics in the CUDA programming model, such as shared memory, unified memory and CUDA streams. Finally, we show how to profile and optimize the given CUDA code.

## Shared Memory Matrix Opertaions

Optimized matrix operations (using shared memory and tiled matrix concept)

## Shared Memory Matrix Opertaions

Here will see how to use the shared memory from the GPUs, which has a good bandwidth within the GPUs compare to access to the global memory. 

We can try two options with the shared memory for the previous ([matrix multiplications](https://github.com/ezhilmathik/PRACE-MOOC-GPU/blob/main/PRACE-MOOC/Week2/Day5/Matrix-Multiplication.cu)) GPU implementation. They are:

* In these examples we will just use a single row to store them in the shared memory.  By doing this single row value will be having very fast access for many columns of the other matrix to compute a single-row product matrix. 

~~~bash
__global__ void matrix_mul(float *a, float* b, float *c,
                                  int width)
{
 __shared__ float aTile[TILE_DIM][TILE_DIM];

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  float single_entry = 0.0f;
  aTile[threadIdx.y][threadIdx.x] = a[row*TILE_WIDTH+threadIdx.x];
  __syncwarp();
  for (int i = 0; i < width; i++) 
    {
      single_entry += aTile[threadIdx.y][i]* b[i*width+col];
    }
  c[row*width+col] = single_entry;
}
~~~

* And for another option would be to store all two entries of row and column of the a and b matrix in the shared memory. 

~~~bash
__global__ void matrix_mul(float *a, float* b, float *c, int width)
{
  __shared__ float aTile[TILE_DIM][TILE_DIM];
  __shared__ float bTile[TILE_DIM][TILE_DIM];

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
    
  aTile[threadIdx.y][threadIdx.x] = a[row*TILE_WIDTH+threadIdx.x];    
  bTile[threadIdx.y][threadIdx.x] = b[threadIdx.y*width+col];

  __syncthreads();

  float single_entry = 0.0f;    
  for (int i = 0; i < width; i++) 
    { 
      single_entry += aTile[threadIdx.y][i]* bTile[i][threadIdx.x];    
    }   
  c[row*width+col] = single_entry;
}
~~~

### Block matrix multiplication (Tiled matrix multiplication)

~~~bash
__global__ void matrix_mul(const float *d_a, const float *d_b, 
			   float *d_c, int width)
{
  __shared__ int a_block[TILE_WIDTH][TILE_WIDTH];
  __shared__ int b_block[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = by * TILE_WIDTH + ty;
  int col = bx * TILE_WIDTH + tx;

  float single_entry = 0;

  for(int i = 0; i < width / TILE_WIDTH; ++i)
    {
      a_block[ty][tx] = d_a[row * width + (i * TILE_WIDTH + tx)];
      b_block[ty][tx] = d_b[col + (i * TILE_WIDTH + ty) * width];
	 
      __syncthreads(); 
	  
      for(int j = 0; j < TILE_WIDTH; ++j)
	{
	  single_entry += a_block[ty][j] * b_block[j][tx];    
	}
      __syncthreads();
    }
  d_c[row*width+col] = single_entry;
}
~~~

## Day 2

## Unified Memory

Unified Memory

## Unified Memory Concept

### Unified Memory 

CUDA unified memory allows a `managed` memory address space that can be shared between CPU and GPU. Currently, this facility is available only on Linux OS systems. Here the data transfer is still happening. But this functionality makes it simpler for the programmer to avoid `cudaMemcpy()` CUDA API calls in the applications; Figure 1 shows the CUDA unified memory approach's simplified concept. 

![alt text](https://drive.google.com/uc?export=view&id=1yWcgJwhfxGiOD3Q4GlUxGB0oSAQTwFw7)
*Figure 1: CUDA unified memory concept*

This is implemented in two ways: setting `cudaMallocManaged()` CUDA API from the host call and setting up globally as `__managed__`. Now we see how to use the unified memory concept in the CUDA application. For this, we will see the two examples, one without unified memory and another one with unified memory. 

#### Without unified memory concept:
To make a clear distinction between the without unified memory and with unified memory, we will show the following steps followed by a coding example. 

1) Allocate the host memory

2) Allocate the device memory 

3) Initialize the host value

4) Transfer the host value to device memory location 

5) Do the computation using the CUDA kernel 

6) Transfer the data from the device to host 

7) Free device memory 

8) Free host memory 

##### Simple example for without unified memory: 

~~~cuda 
//-*-C++-*-
#include "stdio.h"

__global__ 
void AplusB(int *Vector, int a, int b) 
{
  Vector[threadIdx.x] = Vector[threadIdx.x] 
    + a + b + threadIdx.x;
}

int main() 
{
  int N = 100;

  // Allocate the host memory 
  int *Host_Vector = (int *)malloc(N * sizeof(int));

  // Allocate the device memory 
  int *Device_Vector;
  cudaMalloc(&Device_Vector, N * sizeof(int));
  
  // Initialize the host value
  for(int i = 0; i < N; i++)
    Host_Vector[i] = 100;

  // Transfer the host value to device memory location 
  cudaMemcpy(Device_Vector, Host_Vector, N * sizeof(int), 
	     cudaMemcpyHostToDevice);
 
  // Do the computation using the CUDA kernel
  AplusB<<< 1, N >>>(Device_Vector, 10, 100);

  // Transfer the data from the device to host
  cudaMemcpy(Host_Vector, Device_Vector, N * sizeof(int), 
	     cudaMemcpyDeviceToHost);

  for(int i = 0; i < N; i++)
    printf("%d: A+B = %d\n", i, Host_Vector[i]); 

  // Free the device memory 
  cudaFree(Device_Vector); 

  // Free the host memory 
  free(Host_Vector);

  return 0;
}
~~~

#### With unified memory concept:
The following steps show the concept for the unified memory followed by an example. 
As we can notice here, the unified memory brings down from 8 steps into just 4 steps. So basically, we have removed the 2 memory copy and 1 memory allocation and deallocation. But it needs an additional call, that is `cudaDeviceSynchronize()`. It is needed since `cudaMemcpy()` calls are synchronized, in that case, we do not need `cudaDeviceSynchronize()` whereas, for the unified memory concept, we need to use the `cudaDeviceSynchronize()` after the kernel call before we safely use the data from the host. 

2) Allocate the **unified** memory

3) Initialize the value

5) Do the computation using the CUDA kernel 

7)  Free **unified** memory


~~~cuda
//-*-C++-*-
#include "stdio.h"

__global__ 
void AplusB(int *Vector, int a, int b) 
{
  Vector[threadIdx.x] = Vector[threadIdx.x] 
    + a + b + threadIdx.x;
}

int main() 
{
  int N = 100;

  // Allocate the unified memory 
  int *Unified_Vector = (int *)malloc(N * sizeof(int));
  cudaMallocManaged(&Unified_Vector, N * sizeof(int)); 
             
  // Allocate the device memory 
  // int *Device_Vector;
  // cudaMalloc(&Device_Vector, N * sizeof(int));
  
  // Initialize the host value
  for(int i = 0; i < N; i++)
    Unified_Vector[i] = 100;

  // Transfer the host value to device memory location 
  // cudaMemcpy(Device_Vector, Host_Vector, N * sizeof(int), 
  //	     cudaMemcpyHostToDevice);
 
  // Do the computation using the CUDA kernel
  AplusB<<< 1, N >>>(Unified_Vector, 10, 100);

  // Synchronize the kernel call 
  cudaDeviceSynchronize();

  // Transfer the data from the device to host
  // cudaMemcpy(Host_Vector, Device_Vector, N * sizeof(int), 
  // 	     cudaMemcpyDeviceToHost);

  for(int i = 0; i < N; i++)
    printf("%d: A+B = %d\n", i, Unified_Vector[i]); 

  // Free the unified memory 
  cudaFree(Unified_Vector); 

  // Free the host memory 
  // free(Host_Vector);

  return 0;
}
~~~

###  Prefetch Pageable Memory

The Nvidia CUDA facilitates the prefetch pageable memory option from CUDA 8.0 and from the Pascal Nvidia GPU architecture. Migrate data to the destination device and overlap with compute update page table. It has a much lower overhead than page fault in kernel Asyncoperation that follows CUDA stream semantics. Moreover, the memory can be allocated more than the GPU memory size, and Figure 1 shows the schematic view of the prefetch pageable memory option.  

![alt text](https://drive.google.com/uc?export=view&id=1hJtl_OEZa-eZ5nE9GFPuK_scopxUV3o7)
*Figure 2: *

The blow code shows the example of using the `cudaMemPrefetchAsync()`, by doing this gives faster performance compared to just using the `cudaMallocManaged()`

~~~cuda
//-*-C++-*-
#include "stdio.h"

__global__ 
void AplusB(int *Vector, int a, int b) 
{
  Vector[threadIdx.x] = Vector[threadIdx.x] 
    + a + b + threadIdx.x;
}

int main() 
{
  int N = 100;

  // Allocate the host memory 
  int *Unified_Vector = (int *)malloc(N * sizeof(int));
  cudaMallocManaged(&Unified_Vector, N * sizeof(int)); 
             
  // Initialize the host value
  for(int i = 0; i < N; i++)
    Unified_Vector[i] = 100;

  int device = -1;
  cudaGetDevice(&device);

  // pageble memory copy from host to device 
  cudaMemPrefetchAsync(Unified_Vector, N*sizeof(float), 
		       device, NULL);

  // Do the computation using the CUDA kernel
  AplusB<<< 1, N >>>(Unified_Vector, 10, 100);
  
  // pageble memory copy from device to host
  cudaMemPrefetchAsync(Unified_Vector, N*sizeof(float), 
		       cudaCpuDeviceId, NULL);

  // Free the device memory 
  cudaFree(Unified_Vector); 

  return 0;
}
~~~


## CUDA Streams

CUDA Streams

### CUDA Streams 

The CUDA calls are either synchronous or asynchronous. That is, either they enqueue work and wait for the completion or enqueue work and return immediately. However, host and kernel calls are asynchronous, and they overlap between them. However, using the CUDA streams makes the kernel call asynchronous; kernel calls can overlap. This kind of operation is quite helpful when computation and data transfer are happening at the
same time in the application. The CUDA streams in the CUDA programming environment can be created and destroyed as follows: The CUDA streams can be created and destroyed as follows:

~~~bash
// initialize the cuda stream
cudaStream_t stream;
// create the cuda stream
cudaStreamCreate(&stream);
// destroy the cuda stream
cudaStreamDestroy(stream);
~~~


![figure](https://drive.google.com/uc?export=view&id=1Zio6eyGAoGBBLjmTnkrZ3dykIqIH30BB)
*Figure 1: Nvidia CUDA Streams*


The CUDA streams should be passed as an actual argument in the kernel call from the host, for example, `MyKernel<<<126, 512, 0, stream>>>()`. Figure 1 shows a simple schematic example of CUDA streams performing computation and data transfer in parallel. As can be seen, the usage of CUDA streams results in improved performance. The following code listing shows a simple example of using the CUDA streams for
computation and communication in a CUDA application. In this example, `stream1` and `stream2` can be executed concurrently.


~~~bash
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);
computation<<<blocks,threads,0,stream1>>>();
halo_computation<<<blocks,threads,0,stream2>>>();
cudaStreamDestroy(stream1);
cudaStreamDestroy(stream2);
~~~


## Day 4

## CUDA Application Profiling

In this sections, we will see how to profile to Nvidia CUDA code. GPUs have many cores, and different memory options and profiling will help write an optimised GPU code.

## CUDA Code Profiling

### CUDA  Profiling 

This subsection explains how to profile and optimize code that runs on the Nvidia GPUs involving hybrid programming that uses both CPU and Nvidia GPUs. GPUs have many cores and different memory options. In order to write an optimized code on the GPU, it should be profiled and tuned. Nvidia provides four kinds of profiling options for CUDA, OpenACC and OpenMP models. They are [nvprof](https://docs.nvidia.com/cuda/profiler-users-guide/index.html#nvprof-overview), [Visual Profiler](https://docs.nvidia.com/cuda/profiler-users-guide/index.html#getting-started), [NVIDIA Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) and [NVIDIA Nsight Compute](https://docs.nvidia.com/nsight-compute/2020.3/NsightComputeCli/index.html). Among these, nvprof and Visual Profiler will be deprecated and no longer will support the future Nvidia GPUs. The NVIDIA
Volta platform is the [last architecture](https://docs.nvidia.com/cuda/profiler-users-guide/index.html#nvprof-overview) that can support both nvprof and Visual Profiler.

### CUDA Command-Line Profiling

nvprof is a command-line profiling tool from Nvidia, which supports both events (an event is a countable activity, action, or occurrence on a device) and metrics (a metric is a characteristic of an application calculated from one or more event values) from the application. These events and metrics list can be queried by `nvprof --query-events` and `nvprof --query-metrics`. The `nvprof` will collect the events, metrics, timeline
of CUDA-related activities on both CPU and GPU. It mainly collects information on kernel execution, memory transfer, memory set, and other CUDA API calls. The profiled result can be viewed from the console or saved (`--log-file`) in an output file that can be opened in `Visual Profiler`. By default, `nvprof` will profile the entire application. However, sometimes, there is no need for profiling the entire application, and only specific application regions can be profiled. For doing that, an application should have a `cuProfilerStart()` and `cuProfilerStop()` and should have the header files `cuda_profiler_api.h` or `cudaProfiler.h`. The below example shows how to profile just kernel call in the CUDA application.

~~~bash
// cuda profile starts
cuProfilerStart()
// Execute the kernel
vecAdd<<<gridSize, blockSize>>>();
// cuda profile ends
cuProfilerStop()
~~~

Furthermore, during the execution time, nvprof can be set to `nvprof --profile-from-start off`. This specification disables the entire code profiling for a given CUDA application. The example output below illustrates a sample output of entire code profiling using `nvprof` for a vector addition CUDA application. The following command list shows the compilation and profiling of the CUDA code:

~~~bash
# Compilation
$ nvcc -arch=sm_70 Vector-Addition.cu -o Vector-Addition
# Code execution
$ nvprof ./Vector-Addition
~~~

Example output:

~~~bash
==80299== Profiling application: ./vector-add
==80299== Profiling result:
Type
Time(%)      Time Calls       Avg       Min       Max  Name
GPU activities:
 59.89%  7.1680us     2  3.5840us  3.5840us  3.5840us  [CUDA memcpy HtoD]
 24.33%  2.9120us     1  2.9120us  2.9120us  2.9120us  [CUDA memcpy DtoH]
 15.78%  1.8880us     1  1.8880us  1.8880us  1.8880us  vector_add(float*,
                                                       float*, float*, int)
API calls:
 99.47%  212.96ms     3  70.985ms  2.4710us  212.95ms  cudaMalloc
  0.20%  422.01us    97  4.3500us     111ns  164.06us  cuDeviceGetAttribute
  0.16%  341.11us     1  341.11us  341.11us  341.11us  cuDeviceTotalMem
  0.10%  205.20us     3  68.398us  4.2400us  187.24us  cudaFree
  0.04%  79.783us     3  26.594us  17.353us  32.998us  cudaMemcpy
  0.02%  51.755us     1  51.755us  51.755us  51.755us  cudaLaunchKernel
  0.02%  35.662us     1  35.662us  35.662us  35.662us  cuDeviceGetName
  0.00%  7.1410us     1  7.1410us  7.1410us  7.1410us  cuDeviceGetPCIBusId
  0.00%  1.3980us     3     466ns     118ns     957ns  cuDeviceGetCount
  0.00%     566ns     2     283ns     171ns     395ns  cuDeviceGet
  0.00%     198ns     1     198ns     198ns     198ns  cuDeviceGetUuid
~~~

#### MultiGPU Profiling:

Nowadays, it is quite common to use multiple GPUs on a single compute node. In this situation, a multiple GPU CUDA code also can be profiled using `nvprof --print-summary-per-gpu ./a.out`, which will give profiling information for each GPU. The `nvprof --print-gpu-trace ./a.out` can be used for querying a detailed report of each GPU activity. Figure 1 shows the GPU trace for vector multiplication code. Here, we can see the information about the time taken for the CUDA API calls, kernel calls, the number of registers used and shared/dynamic memory allocated per thread block.


![alt text](https://drive.google.com/uc?export=view&id=1cKfjyebNQzkWcuo4wNfmWk-8uNv8L7hp)
*Figure 1: GPU-Trace mode*


Similarly, CPU activities can also be profiled using `nvprof --print-api-trace ./Vector-Addition`. As can be seen in the example below, this will show the trace of CUDA API calls from the CPU.

~~~bash
==84665== Profiling application: ./vector-add
==84665== Profiling result:
   Start  Duration  Name
17.181ms  6.7400us  cuDeviceGetPCIBusId
65.366ms     757ns  cuDeviceGetCount
65.368ms     119ns  cuDeviceGetCount
65.655ms     338ns  cuDeviceGet
65.656ms     237ns  cuDeviceGetAttribute
65.697ms     299ns  cuDeviceGetAttribute
65.708ms     203ns  cuDeviceGetAttribute
65.888ms     336ns  cuDeviceGetCount
65.889ms     128ns  cuDeviceGet
65.891ms  36.109us  cuDeviceGetName
65.928ms  323.78us  cuDeviceTotalMem
66.252ms     218ns  cuDeviceGetAttribute
.
.
66.712ms     169ns  cuDeviceGetUuid
66.721ms  219.40ms  cudaMalloc
286.12ms  4.4720us  cudaMalloc
286.12ms  2.5790us  cudaMalloc
286.16ms  30.096us  cudaMemcpy
286.19ms  17.777us  cudaMemcpy
286.21ms  40.939us  cudaLaunchKernel (vector_add(float*, 
                            float*, float*, int) [112])
286.25ms  31.490us  cudaMemcpy
286.49ms  14.906us  cudaFree
286.51ms  4.3960us  cudaFree
286.51ms  165.13us  cudaFree
~~~


### CUDA Visual Profiling

Nvidia Visual is a GUI profiler, which supports CUDA application events and traces. It provides programmers with good insight into understanding the application's compute time, memory, and memory transfer. The below example shows the CUDA code compilation and how to create an input file for the Visual Profiler.

~~~bash
// CUDA kernel. Each thread takes care of one element of c
__global__ void vecAdd(double *a, double *b, double *c, int n)
{
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
 
    // Make sure we do not go out of bounds
    if (id &lt; n)
        c[id] = a[id] + b[id];
}
.
.
~~~

Compilation:

~~~bash
# Compilation
nvcc -arch=sm_70 Vector-Addtion.cu -o Vector-Addition
~~~

To create the input file for the profiling:

~~~bash
# To create a .nvvp file for the visual profiler
nvprof -o input.nvvp ./Vector-Addition
~~~

To open the Visual Profiler, use:

~~~bash
nvvp input.nvvp
~~~


and input file should have a .nvvp; it can be created as `nvprof -o input.nvvp ./a.out`, as shown above. Figure 2 shows the timeline of the CUDA API in the application. It shows all the CUDA API calls time consumption for the given application. Especially, these calls are shown in the total timeline of the application. Figure 2 is not only showing the CUDA API calls but also shows the time consumption for the CUDA kernel calls. The timeline will give a clearer overview of where the application spends most of the time.


![alt text](https://drive.google.com/uc?export=view&id=1fqCYaiFMfiHNdW9npul-qxqVDKwpSyvV)
*Figure 2: GPU-Trace mode*


### Nsight Compute and Nsight Systems

The next generation of Nvidia GPUs will only support [Nsight Compute and Nsight Systems](https://docs.nvidia.com/cuda/profiler-users-guide/index.html#visual). Both of these profiling tools will help to profile the GPUs and CPU-GPU together. Figures 3 and 4 illustrate the difference between the regular Nvidia profiling and Nsight Compute. Here, during the regular (traditional) profiling, the user will execute the CUDA application; it will communicate with the CUDA runtime library and CUDA driver. But in the Nsight Compute, the CUDA application will be executed by Nsight Compute.



![alt text](https://drive.google.com/uc?export=view&id=1ca-wvg-gNMpP1_uEUfWqM4uqHSgG2EG3)
*Figure 3: Regular CUDA Profiling*



![alt text](https://drive.google.com/uc?export=view&id=1vqiaD-h7NoknzSGFrj9hKFWOnbq4-HXK)
*Figure 4: Nvidia Nsight Compute*



The `ncu` will be used as a profiling command to collect the default options (events and metrics) in the application, for example, using the command: `ncu -o profile ./a.out`. The profiled result can be seen in the command line temporarily but also using the flag `-o`, with `-o profile` will create `profile.ncu-rep`, which will have all the profiled result. The example below shows the profiling results of two CUDA kernel calls in the application.

~~~bash
[Vector addition of 1144477 elements]
==PROF== Connected to process 5268
Copy input data from the host memory to the CUDA device
CUDA kernel launch A with 4471 blocks of 256 threads
==PROF== Profiling "vectorAdd_A" - 1: 0%....50%....100% - 46 passes
CUDA kernel launch B with 4471 blocks of 256 threads
==PROF== Profiling "vectorAdd_B" - 2: 0%....50%....100% - 46 passes
Copy output data from the CUDA device to the host memory
Done
==PROF== Disconnected from process 5268
==PROF== Report: profile.ncu-rep
~~~

Applications using multiple GPUs on a single node can be also profiled using `ncu--target-processes all -o <single-report-name> <app> <args>`. Nsight Compute can support GUI as well; to invoke GUI use: `ncu-ui <profile.ncu-rep>`. The Nvidia Nsight compute provides many metrics, to see the list of metrics, use:`--query-metrics`. However, sometimes, it would be just easier to collect and analyse the set of metrics. For doing that, Nsight provides sets and sections. To see the list of sets use: `ncu --list-sets` and to query the current list of sections use: `ncu --list-sections`. By default, the sections are associated with the default set.

Nsight Systems is also able to profile the CUDA applications. A command line argument for Nsight System is `nsys`. To profile the application, use `nsys profile ./a.out`, which will profile the entire application with default events and metrics; `nsys --help` will list out the available options in `nsys`.
## Day 5

## Performance Optimization and Tuning

In this section, we show a few performance tuning options for CUDA programming. 


## Performance Optimization and Tuning
### Performance Optimization and Tuning 

#### CUDA Occupancy Calculator 

CUDA occupancy is defined as a ratio of the active warps to a maximum amount of the warps that can support the given Nvidia architecture. Each Nvidia architecture has an N number of registers, and these are shared among the thread blocks. The CUDA compiler tries to minimize the number of registers per thread blocks to maximize the CUDA occupancy. However, these thread blocks and register usage differ to different compute capability and Nvidia architecture. To make these combinations (registers and thread blocks) easy, Nvidia provides the preprepared simple calculation of the [CUDA Occupancy Calculator[XLS]](https://docs.nvidia.com/cuda/cuda-occupancy-calculator/index.html) sheet to the users. With this, users can choose an optimized thread block depending on the compute capability and Nvidia architecture in the CUDA
kernel without trying many combinations of thread blocks by themselves.

#### Register Usage

Minimizing the register usage helps to launch a maximum number of warps in the streaming multiprocessor, see Figure 1. However, restricting the register usage leads to the register spilling to the L1 cache; for instance, spilling fewer registers, warps waiting time will not be decreased, and spilling more registers, communication time between L1 cache and registers will increase. So several test cases should be done before choosing the appropriate register number according to the problem. This is usually achieved in two ways: using the flag `-maxrregcount=<N>` and `__launch_bounds__`. Here, `N` is the number of registers that can be allocated per warps. The Code syntax below shows how to control the register usage with `__launch_bounds__`.

~~~cuda
#define THREADS_PER_BLOCK          256
#if __CUDA_ARCH__ >= 200
    #define MY_KERNEL_MAX_THREADS  (2 * THREADS_PER_BLOCK)
    #define MY_KERNEL_MIN_BLOCKS   3
#else
    #define MY_KERNEL_MAX_THREADS  THREADS_PER_BLOCK
    #define MY_KERNEL_MIN_BLOCKS   2
    #endif

// Device code
__global__ void
__launch_bounds__(MY_KERNEL_MAX_THREADS, MY_KERNEL_MIN_BLOCKS)
MyKernel(...)
{
    ...
}
~~~


![figure](https://drive.google.com/uc?export=view&id=1KGPXvUqBemR24UjMyPx8ziKUR4YBR1DG)
*Figure 1: Schematic diagram of warps are being executed in the streaming multiprocessors*


#### Cache Configuration
      
Depending on the CUDA application, L1 cache and shared memory sizes can be modified by using CUDA runtime `cudaDeviceSetCacheConfig()` API call. Registers hold the frequently used values in the computation, but when the registers are more than the actual available count, it spills into different memory levels. This kind of behaviour will slow down the performance of an application. Nvidia GPUs have an L1 cache and shared memory, so increasing the L1 cache size will accommodate the spilt registers and not let them go further into shared memory or global memory. Depending on the application, it might not be needed to use all the registers but to use the shared memory more. In that case, shared memory size can be increased, and the L1 cache size can be decreased. The list below shows the example usage of `cudaDeviceSetCacheConfig()` in the CUDA programming environment. 

~~~cuda
// example usage
cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

// Default function cache configuration, no preference
cudaFuncCachePreferNone
// Prefer larger shared memory and smaller L1 cache
cudaFuncCachePreferShared
// Prefer larger L1 cache and smaller shared memory
cudaFuncCachePreferL1

// simple example usage increasing more shared memory 
#include<stdio.h>;
int main()
{
    // example of increasing the shared memory 
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
    My_Kernel<<<>>>();
    cudaDeviceSynchronize(); 
    return 0;
}
~~~


#### Memory Access Through NVLink Interconnect


NVLink from Nvidia is a high-speed data interconnect. Using NVLink, a data transfer between the GPUs within the single compute node can be increased than using a data transfer via CPU. NVLink completely bypasses the CPU for the data transfer between GPUs. Figure 2 shows the simple schematic overview of NVLink interconnects in the Nvidia GPUs. For example, the GV100 can support up to six NVLink interconnections. Each has up to 50GB/s of bi-directional bandwidth. Nvidia Volta can similarly support up to six NVLink with a total bandwidth of 300GB/s.


![figure](https://drive.google.com/uc?export=view&id=1irNFRDb7ju1fzxdpU2altnIeEXxbcYEt)
*Figure 1: NVLink GPU-to-GPU connections*

The NVLink capability should be enabled via `cudaDeviceEnablePeerAccess()` and `cudaMemcpyPeer()` API calls. The example below shows how to enable Peer-to-Peer Memory Access using `DeviceEnablePeerAccess()`, which can directly access the Vector allocated memory from `Device 0` in `Device 1`.

~~~cuda
cudaSetDevice(0);                     // Set device 0 as current
float* Vector;
size_t size = 1024 * sizeof(float);
cudaMalloc(&Vector, size);            // Allocate memory on device 0
MyKernel<<<128, 128>>>(vec);          // Launch kernel on device 0
cudaSetDevice(1);                     // Set device 1 as currentcuda
DeviceEnablePeerAccess(0, 0);         // Enable peer-to-peer access
                                      // with device 0
// Launch kernel on device 1
// This kernel launch can access memory on device 0 at address vec
MyKernel<<<128, 128>>>(Vector);
~~~


The example below illustrates the demo of the peer-to-peer memory copy using `cudaMemcpyPeer()`. Here, an allocated memory on `Device 0` can be copied into `Device 1`.


~~~cuda
cudaSetDevice(0);                      // Set device 0 as current
float *Vector0;
size_t size = 1024 * sizeof(float);
cudaMalloc(&Vector0, size);            // Allocate memory on device 0
cudaSetDevice(1);                      // Set device 1 as current
float *Vector1;
cudaMalloc(&Vector1, size);            // Allocate memory on device 1
cudaSetDevice(0);                      // Set device 0 as current
MyKernel<<<128, 128>>>(Vector0);       // Launch kernel on device 0
cudaSetDevice(1);                      // Set device 1 as current
// Copy Vectoro to Vector1
cudaMemcpyPeer(Vector1, 1, Vector0, 0, size); 
~~~

#### Nvidia GPUDirect
Nvidia GPUDirect is a part of [Magnum_IO](https://www.nvidia.com/en-us/data-center/magnum-io/) technology. It will enable faster data movement in the GPUs. It has two functionality: [GPUDirect Storage](https://docs.nvidia.com/gpudirect-storage/overview-guide/index.html) and [GPUDirect Remote Direct Memory Access (RDMA)](https://docs.nvidia.com/cuda/gpudirect-rdma/index.html). Usage of this functionality will eliminate unnecessary memory copy, decrease CPU overheads and reduce latency. GPUDirect Storage can be used to transfer data from external storage to GPUs. GPUDirect RDMA enables faster data transfer between the GPUs across the different compute nodes. Figures 3 and 4 show the simple illustration of these two methodologies. For more information regarding Nvidia GPUDirect, please refer to [Nvidia GPUDirect documentation](https://developer.nvidia.com/gpudirect).


![figure](https://drive.google.com/uc?export=view&id=1m7wLTgNXoi7m_IG4_LKB8XuqTeEMorsf)
*Figure 3: Nvidia GPUDirect Storage*


![figure](https://drive.google.com/uc?export=view&id=1jwJYbzxh2-9qSiLfn4eobhqC9XczJuV0)
*Figure 4: Nvidia GPUDirect RDMA*








