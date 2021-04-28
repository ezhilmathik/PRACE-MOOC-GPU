## Part 1

Presently there are many [GPU vendors](https://www.jonpeddie.com/reports/market-watch/)
available in the market for GPU architecture, in particular,
`AMD`, `Nvidia`, and `Intel`. But Nvidia shares a large percentage.
Therefore, we focus on GPU programming for Nvidia.
Compute Unified Device Architecture (CUDA)
is a programming language that uses to program the Nvidia GPUs. 

![alt text](https://drive.google.com/uc?export=view&id=1Vcc4x9mGfK-iJHuhJ49-fGCsi1MAya0_)
*[source: https://jonpeddie.com](https://www.jonpeddie.com/store/market-watch-quarterly1)*

### Why learning GPU programming:

* Each and every year supercomputers computational power is increasing.
* [The topmost 10 supercomputers have GPU as accelerators](https://www.top500.org/).
* To be able to use available resources from the hardware side, we should be 
  able to run our scientific codes on parallel architecture. 

![alt text](https://drive.google.com/uc?export=view&id=1ANhLrLAmeHdIyE4ysyZnqQA449Fgq8tu)
*[source: https://www.top500.org/](https://www.top500.org/)*


### Things you will learn from this course:

* GPU architecture (memory hierarchy, streaming multiprocessor, TPU, etc.).
* GPU parallel programming (Using CUDA and OpenACC).
* GPU programming for computational numerical linear algebra.
* Code optimization & profilling (fine-tuning and profiling technique).


## Part 2

### GPU architecture

* GPU has many cores: In general GPU has many cores compare
  to typical CPUs. But on the other hand frequency of the
  CPU is higher than the GPUs. That makes CPU faster in
  computing compare to GPU but still, GPUs win because,
  it can handle many threads, that can process many data in parallel. 

![figure](https://drive.google.com/uc?export=view&id=1khagSDq91t7uZipGgPCM-Nf37yYFEraJ)
*Figure 1: CPU vs GPU archietecture* [source](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#introduction)


* In the GPU, cores are grouped into `GPU Processing Clusters`
  (GPCs) and each GPCs has its own `Streaming Multiprocessors
  (SMs)` and `Texture Processor Clusters (TPCs)`. 

![figure](https://drive.google.com/uc?export=view&id=1RRDH7Wfaz8Vo3ueJl-mg1XW83Wd5QwaY)
*Figure 1: GPCs, SMs and TPCs* [source](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#introduction)


* Streaming multiprocessors in the GPU are capable of running
  multiple threads concurrently; this is called a `Single
  Instruction Multiple Thread (SIMT)` architecture.
  SIMT enables programmers to achieve thread level parallelism
  by using a single instruction with multiple threads that can handle multiple data. 

* Each streaming multiprocessor has their own instruction cache, registers, shared
  memory, L1 cache, constant cache and texture cache; L2 cache, constant memory,
  texture memory and global memory are shared among the multiple streaming multiprocessor.

* SMs in the GPU are based on the scalable array multi-thread,
  which allows grid and thread blocks of `1D`, `2D`, and `3D` data.
  Therefore, programmers can write the grid and block size to
  create a thread when executing the device kernel; this thread
  block is called a `Cooperative Thread Array (CTA)`. GPU performance can be
  improved if the **latency** is hidden; latency is the number of clock cycles needed
  to execute the next warp in the SM. Each SM has `single-precision CUDA cores`,
  `double-precision units`, `special function units`, and `load/store units`.

* A parallel execution is happening in the SMs and also in the **warps**.
  One warp contains `32 threads`; warps can spawn across the SMs,
  and each warp has their own instructions and registers.
  The multiprocessor occupancy is the ratio of active warps
  to the maximum number of warps supported on a multiprocessor of
  the GPU. The Nvidia has a predetermined the
  [CUDA Occupancy Calculator](https://docs.nvidia.com/cuda/cuda-occupancy-calculator/index.html)
  different Nvidia GPU architecture. 


### Nvidia Microarchitecture
* The Nvidia microarchitecture frequently releases the updated
  version of their GPUs microarchitecture. The following
  list shows some of their architectures: Tesla (2006),
  Fermi (2010), Kepler (2012), Maxwell (2014),
  Pascal (2016), Volta (2017), Turing (2018), and Ampere (2020) 
* Each and every updated architecture have more advanced
  features compared to their previous releases. 
* There are different GPUs that are available
  based on each and every `Nvidia microarchitecture`.


### [Compute capabilities](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability)
* Nvidia GPUs features can be explored by programmers by
  using the `compute capabilities`. These compute capabilities
  determine the different functions that can be used by the programmer.
  Each and every Nvidia microarchitecture has it's own compute
  capabilities, and usually, they are evoked in the
  compilation time as a compilation flag <code> -arch=compute_70</code>.
  For example, Tensor cores, half-precision, and mixed-precision
  functionality are achieved by specifying the compute capabilities. 


### [Unified Memory](https://developer.nvidia.com/blog/unified-memory-cuda-beginners/)

* From Maxwell Nvidia, GPUs support a unified memory approach,
  which means both GPU and CPU can access read,
  and write the same address with simplified
  memory allocation while you writing the programme.

![figure](https://drive.google.com/uc?export=view&id=1s3PbSwz5nRxSBh9xdEZ29bwSZQe7Pn2d)
*Figure 2: Unified memory* [source](https://developer.nvidia.com/blog/unified-memory-cuda-beginners/)

### [Nvidia NVLink](https://www.nvidia.com/en-us/design-visualization/nvlink-bridges/) 

* `NVLink` is the latest technology from the Nvidia that connects
  the multiple GPU, which would transfer the data between
  the GPUs much faster compared to traditional PCIe-based solutions.
  Figure 3 shows how the GPUs are connected together using
  the [NVSwitch and NVLink](https://www.nvidia.com/en-us/data-center/nvlink/).
  And Figure 4 shows the data transfer performance in different Nvidia microarchitecture.     

![figure](https://drive.google.com/uc?export=view&id=1rVnHoiV_EMu_0ivrsEasc94UfeVgRXnb)
*Figure 3: Nvidia NVSwitch* [source](https://www.nvidia.com/en-us/data-center/nvlink/)

![figure](https://drive.google.com/uc?export=view&id=1IoVsFEDfP2kFaTOGvDe8flPncBmxPSOD)
*Figure 4: Nvidia NVLink* [source](https://www.nvidia.com/en-us/data-center/nvlink/)

### Understanding Your GPUs

* Most of the time, when we run our GPU (CUDA) code,
  we do not know exactly what is the Nvidia GPU type we run.
  Understanding your GPU card will give your information 
  about grid, block, architecture and memory, 
  that will be useful for writing optimized and efficient code,
  see [CUDA Device Query](https://raw.githubusercontent.com/ezhilmathik/PRACE-MOOC-GPU/main/code/query.cu?token=AK3BVJQBLNJEOK2PQBQ2B5DAISNEK). 

~~~bash 
CUDA Device Query...
There are 1 CUDA devices.

CUDA Device #0
Major revision number:         7
Minor revision number:         0
Name:                          Tesla V100-SXM2-16GB
Total global memory:           4060610560
Total shared memory per block: 49152
Total registers per block:     65536
Warp size:                     32
Maximum memory pitch:          2147483647
Maximum threads per block:     1024
Maximum dimension 0 of block:  1024
Maximum dimension 1 of block:  1024
Maximum dimension 2 of block:  64
Maximum dimension 0 of grid:   2147483647
Maximum dimension 1 of grid:   65535
Maximum dimension 2 of grid:   65535
Clock rate:                    1530000
Total constant memory:         65536
Texture alignment:             512
Concurrent copy and execution: Yes
Number of multiprocessors:     80
Kernel execution timeout:      No
~~~


## Part 3







