### We will now look into how this CUDA threads are working. 

* It is really important to understand, how the CUDA threads are created and working within the CUDA programming. 

* Within the CUDA programming, threads are organized as 1D, 2D and 3D. We will explain later why we have this methodology in CUDA programming. 

![alt text](https://drive.google.com/uc?export=view&id=12-YID5_NEbgqSUT30bcgN7De_GKwrJKI)
*[source: https://docs.nvidia.com/](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)*


![alt text](https://drive.google.com/uc?export=view&id=1sZghM6BcuHDyyEoqCf-yugG02WdYZVDx)
*[source: https://docs.nvidia.com/](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)*

* The [CUDA occupancy calculator](https://docs.nvidia.com/cuda/cuda-occupancy-calculator/index.html) will provide you with a simple overview of how CUDA threads should be organized. There are pre-computed in [here](https://docs.nvidia.com/cuda/cuda-occupancy-calculator/CUDA_Occupancy_Calculator.xls) for different CUDA architecture. Please note, the preconfigured recommendation, does not work well for all the cases. This is just an overview of how the threads should be organized and simple and standard cases. 



| `Memory`  | `Location` | `Cached` | `Device Access` | `Scope`                | `Life Time`    |
|----------|----------|--------|---------------|----------------------|--------------|
| **Register** | On-chip  | N/A    | R/W           | one thread           | thread       |
| **Local**    | DRAM     | Yes**  | R/W           | one thread           | thread       |
| **Shared**   | On-chip  | N/A    | R/W           | all threads in block | thread block |
| **Global**   | DRAM     | *      | R/W           | all threads in host  | Application  |
| **Constant** | DRAM     | Yes    | R             | all threads in host  | Application  |
| **Texture**  | DRAM     | Yes    | R             | all threads in host  | Application  |

    * cached L2 by default by latest compute capabilities
    ** cached L2 by default only on compute capabilities 5.x

#### Grids, Blocks and Threads organization in CUDA:

* In CUDA, threads are organized as Grids, Blocks, and Threads. And they can be defined as 1D, 2D and 3D blocks. For example, a CUDA kernel function can be called as `VecMul<<<1, N>>>(A, B, C);`
* CUDA also provided the builtin function to define the threads.
    * gridDim.x, gridDim.y and gridDim.z are defining the number of the blocks in the grids in x,y and z dimension.
    * blockIdx.x, blockIdx.y and blockIdx.z are defining the index of the current block within the grid in x, y and z dimension. 
    * blockDim.x, blockDim.y and blockDim.z are defining the number of threads in the block in x, y and z dimension. 
    * threadIdx.x, threadIdx.y and threadIdx.z are defining the index of the thread within a block in x, y and z dimension. 

~~~
// Kernel invocation with one block of N * N * 1 threads
int numBlocks = 1;
dim3 threadsPerBlock(N, N);
MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
~~~

* In some situations, the whole thread should be converted to just a single array of thread. But threads can be created as 1D, 2D and 3D. However, at the same time, in a particular application, we just need a single array in the global function. The below list shows an example of different CUDA dimensions into different dimensions. 

~~~ C++
//1D grid of 1D blocks
__device__ int getGlobalIdx_1D_1D()
{
  return blockIdx.x * blockDim.x + threadIdx.x;
}

//1D grid of 2D blocks
__device__ int getGlobalIdx_1D_2D()
{
  return blockIdx.x * blockDim.x * blockDim.y
    + threadIdx.y * blockDim.x + threadIdx.x;
}

//1D grid of 3D blocks
__device__ int getGlobalIdx_1D_3D()
{
  return blockIdx.x * blockDim.x * blockDim.y * blockDim.z 
    + threadIdx.z * blockDim.y * blockDim.x
    + threadIdx.y * blockDim.x + threadIdx.x;
}

//2D grid of 1D blocks 
__device__ int getGlobalIdx_2D_1D()
{
  int blockId   = blockIdx.y * gridDim.x + blockIdx.x;				
  int threadId = blockId * blockDim.x + threadIdx.x; 
  return threadId;
}

//2D grid of 2D blocks  
 __device__ int getGlobalIdx_2D_2D()
{
  int blockId = blockIdx.x + blockIdx.y * gridDim.x; 
  int threadId = blockId * (blockDim.x * blockDim.y) +
    (threadIdx.y * blockDim.x) + threadIdx.x;
  return threadId;
}

//2D grid of 3D blocks
__device__ int getGlobalIdx_2D_3D()
{
  int blockId = blockIdx.x 
    + blockIdx.y * gridDim.x; 
  int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
    + (threadIdx.z * (blockDim.x * blockDim.y))
    + (threadIdx.y * blockDim.x)
    + threadIdx.x;
  return threadId;
}

//3D grid of 1D blocks
__device__ int getGlobalIdx_3D_1D()
{
  int blockId = blockIdx.x 
    + blockIdx.y * gridDim.x 
    + gridDim.x * gridDim.y * blockIdx.z; 
  int threadId = blockId * blockDim.x + threadIdx.x;
  return threadId;
}

//3D grid of 2D blocks
__device__ int getGlobalIdx_3D_2D()
{
  int blockId = blockIdx.x 
    + blockIdx.y * gridDim.x 
    + gridDim.x * gridDim.y * blockIdx.z; 
  int threadId = blockId * (blockDim.x * blockDim.y)
    + (threadIdx.y * blockDim.x)
    + threadIdx.x;
  return threadId;
}

//3D grid of 3D blocks
__device__ int getGlobalIdx_3D_3D()
{
  int blockId = blockIdx.x 
    + blockIdx.y * gridDim.x 
    + gridDim.x * gridDim.y * blockIdx.z; 
  int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
    + (threadIdx.z * (blockDim.x * blockDim.y))
    + (threadIdx.y * blockDim.x)
    + threadIdx.x;
  return threadId;
}
~~~


For example a simple program prints the thread layout from the GPU function.

~~~bash
//-*-C++-*-
#include <stdio.h>

__global__ void helloCUDA()
{
 // converting 2D thread structure into 1D thread structure 
  int i = blockIdx.x * blockDim.x * blockDim.y 
  + threadIdx.y * blockDim.x + threadIdx.x;
  printf("Hello thread %d\n", i);
  __syncthreads();       
}

int main()
{
  // Thread organization (2D block)
  dim3 comp_G(1, 1, 1);    
  dim3 comp_B(8, 8, 1); 

  helloCUDA<<<comp_G, comp_B>>>();

  cudaDeviceReset();

  return 0;
}
~~~

