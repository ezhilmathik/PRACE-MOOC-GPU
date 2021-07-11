## Week 1: Course Organization, Parallel Programming Concepts and GPU architecture

This week, we go through the introduction to the course, continuing with checking prerequisites and accessing the different ways of the GPU resources (both hardware and software). Finally, we see the basics of parallel programming and a detailed study of GPU architecture. 

## Day 1

## Introduction 
 This article will show the importance of GPU computing in scientific computing and artificial intelligence; and how GPUs are incorporated into modern supercomputers. 
 
 ### GPGPU programming 

In this course, you will be learning GPU programming using Compute Unified Device Architecture (CUDA) and OpenACC. CUDA is a low-level programming language for the Nvidia GPUs, whereas OpenACC can be used for both modern processors and accelerators. Programming on GPUs is generally referred to as General-purpose graphical process unit (GPGPU) programming. GPUs provide higher throughput and bandwidth than CPU; this is one of the main reasons GPUs are suitable for scientific computing. Nowadays, GPUs are used for many applications, from user end applications (e.g. video games) to artificial intelligence and scientific computing (simulations for science and engineering); see Figure 1 and [few applications that use the Nvidia GPUs](https://www.nvidia.com/en-us/gpu-accelerated-applications/). 

![figure](https://drive.google.com/uc?export=view&id=1lYXIC9Fj6cx7MurKyzL9D0UKcfGUoV2_)
*Figure 1: Pictures clockwise:(a) Astrophysics; (b) CFD:turbulence; (c) Bioinformatics; (d) Material science*

Presently, [the topmost 10 supercomputers have GPU as accelerators](https://www.top500.org/) and every year, supercomputers' computational power is increasing, see Figure 2(left) as of November 2020 and Figure 2(right) shows the share of the GPU vendors as of November 2020. Therefore we should run our scientific codes on parallel architecture utilizing both CPU and GPUs to use available resources from the hardware side.  

![figure](https://drive.google.com/uc?export=view&id=1RudzYkUYQbaZ2Vblq35y1VOzd5Qu0j22)
*Figure 2: (left) Computing power; (right) Share of the accelerators*

All the ongoing exascale and pre-exascale projects have accelerators, and for example, this can be seen in Table 1 and refer to Figure 3 for sample supercomputer layout. As we notice here, all the machines listed in Table 1 have either AMD or Nvidia GPUs. This is a clear indication that all scientific application software should utilize heterogeneous computing; otherwise, we will not use the hardware resources. 

|	|[Frontier](https://www.olcf.ornl.gov/frontier/)|[Aurora](https://alcf.anl.gov/aurora)|[El Capitan](https://www.llnl.gov/news/llnl-and-hpe-partner-amd-el-capitan-projected-worlds-fastest-supercomputer)|[Leonardo](https://eurohpc-ju.europa.eu/discover-eurohpc#ecl-inpage-211)|
|--|--|--|--| 
|CPU Architecture|AMD EPYC|Intel Xeon Scalable|AMD EPYC "Genoa" (Zen 4)|Intel Xeon Ice Lake|
|GPU Architecture|[Radeon Instinct](https://www.amd.com/en/graphics/servers-radeon-instinct-mi)|[Intel Xe](https://en.wikipedia.org/wiki/Intel_Xe)|Radeon Instinct|[Nvidia Amphere Architecture](https://www.nvidia.com/en-us/data-center/nvidia-ampere-gpu-architecture/)|
|[Performance (RPEAK)](https://en.wikipedia.org/wiki/FLOPS)|>1.5 EFLOPS|1.0 EFLOPS|2.0 EFLOPS|322.6 PFLOPS|
|Power Consumption|~30 MW|<=60 MW|<40 MW|9 MW|
|Laboratory|[Oak Ridge Leadership Computing Facility](https://www.olcf.ornl.gov/)|[Argonne Leadership Computing Facility](https://www.alcf.anl.gov/)|[Lawrence Livermore National Laboratory](https://www.llnl.gov/)|[CINECA](https://www.cineca.it/en/our-activities/data-center/hpc-infrastructure/leonardo)|
|Vendor|Cray| Intel|Cray|Intel|
|Expected Year|2021|2021|2023|2021|

*Table 1: The ongoing exascale and pre-exascale projects in the USA and Europe.*




![figure](https://drive.google.com/uc?export=view&id=1_c0cxDa5j44LxazP1Cpg5jn6EjYz40WS)

*Figure 3: Frontier and Lumi supercomputers* 

Both in science & engineering and artificial intelligence, we need to do lots of arithmetic computation with numbers, see Figure 4 for application areas of HPC, and GPU vendors share in HPC. For example, in science and engineering, problems are defined by partial differential equations (PDEs). These PDEs are converted into a system of equations by using numerical methods (e.g. finite difference and finite element methods), where we need to find the values for the unknown variables.


![figure](https://drive.google.com/uc?export=view&id=1VxeEbGGmwK5wgcMxyhwVSTwpnPO10LRN)
*Figure 4: (left) Applications uses HPC resources; (right) GPU vendors support HPC*


Similarly, in artificial intelligence, we end up solving matrices and vectors, refer to Figure 5. To do these computations need huge computational power. Using the GPUs, we can achieve faster data parallelization, enabling quicker computation on the GPUs. 


![figure](https://drive.google.com/uc?export=view&id=19dUJnyY3IlE4LO-yl-F2YW14zmtuJBgK)
*Figure 5: Numerical computation arises from computational science and artificial intelligence* 


Programming on GPUs needs special programming language requirements. There are many GPU programming languages existing, namely, CUDA, OpenACC and OpenCL. All of these languages have their own programming structure and syntax. However, in this course, we will only learn the CUDA (applicable only for Nvidia GPUs) and OpenACC (for any GPUs).



## Day 2


### Course Organization and GPU Access

This article gives an overview of the outcome of the course, prerequisite and course structure. Lastly, it suggests ways of accessing the GPU hardware and software. 

### Course organization 

### Prerequisite:

To follow this course, learners would expect to have basic knowledge in C/C++ and Fortran programming language. This course is focusing on CUDA and OpenACC programming language. CUDA programming focuses on the C/C++ programming language, whereas the OpenACC programming model focuses on C/C++ and Fortran programming models. The blow list shows the prerequisites needed for this course.

* [C/C++](https://www.cprogramming.com/tutorial/c-tutorial.html?inl=hp) programming language (compulsory)
    - C/C++ programming language is a very low-level programming language, this means when we do programming, we can have full control of the memory of the system and offer good scalability. 
* [Fortran](https://fortran-lang.org/learn/) programming language (optional)
    - Fortran is a very old language and still popular among the scientific community because most of the compute-intensive numerical simulations were written using Fortran, for example, climate modelling and weather forecasting codes. Plus also it offers good memory control and scalability over the machine with minimal coding effort.

* [OpenMP](https://hpc.llnl.gov/openmp-tutorial) programming language (optional)
     - OpenMP is discussed a little bit in the next article of this course. 


|Model	|Implementation	|Supported languages|	Target architectures|
|--|--|--|--|
|OpenACC|	Directives|	Fortran, C, C++|	CPUs, GPUs, OpenPOWER|
|OpenMP	|Directives	|Fortran, C, C++	|CPUs, Xeon Phi, GPUs|
|CUDA	|Language extension	|(Fortran), C, C++	|GPUs (NVIDIA)|
|OpenCL	|Language extension	|C, (C++)	|GPUs, (CPUs), FPGAs|
|C++ AMP	|Language extension|	C++	|CPUs, GPUs|
|RAJA|	C++ abstraction|	C++	|CPUs, GPUs|
|TBB	|C++ abstraction|	C++	|CPUs|
|C++17	|Language feature|	C++	|CPUs|
|Fortran 2008	|Language feature	|Fortran	|CPUs|



If you do not know those listed programming models, we strongly advise you to learn some basics before continuing this course. 


### Course structure:

* The course is aimed for five weeks
* This course has two parts: one is dedicated to CUDA programming (low-level programming for Nvidia GPUs) from basic to advanced. The second focuses on the OpenACC programming model for any GPU vendors (any accelerators) from basic to advanced. 
* Each week has 5 sections; each section will have one article, quiz and discussion.
* And one video per week will give you a complete overview of that particular week's course content. 


### What will you learn from this course:

* GPU (Nvidia and AMD) architecture (memory hierarchy, streaming multiprocessor, TPU, etc.).
* GPU (CUDA and OpenACC) parallel programming (CUDA-threads organization, OpenACC directives and clauses).
* GPU (CUDA and OpenACC) programming for computational numerical linear algebra.
* Advanced topics in CUDA and OpenACC programming model. 
* Code optimization (CUDA and OpenACC): Profiling technique and fine-tuning. 


### Different ways of getting GPU access:

There are three possible ways you have GPU access. They are your own personal GPU based laptop, you can get GPU access via cluster/supercomputer, and finally, you can access the GPUs via Cloud platform (e.g. Google and Amazon). The following list shows basic ideas and instructions to access the GPUs, compile and run the GPU codes. 

* Personal Computer:
    - This is usually possible when you buy a laptop or desktop, where we get to choose the GPU configuration. To be able to use GPU for scientific computations, you just need to install the compilers, which is explained in the following section. 

* Via Cluster/Supercomputer:
    * Login: To be able to use the small cluster or supercomputer, your local computer should be connected to it. To do that, usually, you have to follow the instructions of the [SSH](https://www.ssh.com/ssh/protocol/) protocol of that local cluster/supercomputer. Usually, you can find this information in their technical documentation (mentioning access or connection). 
    * Load the modules:  Once you have logged in to the cluster/supercomputer, you should load the needed software environment modules. In this case, it should be either 
CUDA toolkit or PGI compiler. Typically, software modules are installed at the cluster/supercomputer via [Lmod](https://lmod.readthedocs.io/en/latest/). After this, you can able to compile your CUDA code or OpenACC code on the cluster/supercomputer. 

* Through Cloud platform:
    * Google: Using the [Google Cloud](https://cloud.google.com/gpu), anyone can have access to the [latest GPU architecture](https://cloud.google.com/compute/docs/gpus) and based on usage of the [GPU the pricing](https://cloud.google.com/compute/gpus-pricing) is determined.  End users also can create their own [Virtual machine](https://cloud.google.com/compute/docs/gpus/create-vm-with-gpus) with desired GPU configuration, and Google Cloud provides the [installation instruction for the CUDA toolkit on the virtual machine](https://cloud.google.com/compute/docs/gpus/install-drivers-gpu). 
    * Amazon: It provides hardware and software access as a Cloud solution; through [customer support](https://aws.amazon.com/contact-us/), more information can be given on accessing the GPU hardware and software in Amazon Cloud. 


### Compiler Requirements:
* CUDA Toolkit (for CUDA programming model):
    - CUDA Toolkit supports Linux, Mac and Windows OS. 
    - For Linux, please refer to [install steps](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).
     - For Windows, please refer to [install steps](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html).
     For Mac, Nvidia CUDA Toolkit's installation can be found [here](https://docs.nvidia.com/cuda/archive/10.2/pdf/CUDA_Installation_Guide_Mac.pdf).

* PGI compiler (for OpenACC programming model):
     - PGI compiler only supports the Linux and Windows OS; for the installation instructions, please refer to [installation documentation](https://www.pgroup.com/resources/docs/20.4/x86/pgi-install-guide/index.htm).


## Day 3

## Parallel Computer Architectures

In this article, we will study the basics of computer parallel architecture. This will give an overview of how the parallel architecture of CPU working. By knowing this, later it would help to understand the GPU architecture. 

### Introduction to Parallel Computer Architectures

Parallel processing concepts existed even in the pre-electronic computing era in the 19th century.  For instance, Babbage (Babbage’s analytical engine, 1910 ) considered parallel processing for speeding up the multiplication of two numbers by using his difference engine.

The Von Neumann architecture helps to provide sequential architecture at the beginning of high-speed computing.  Figure 1 (a)  illustrates the  Von  Neumann architecture.  Even though the computation may have been done very fast, there is a limitation with I/O to the memory is called Von Neumann bottleneck (Alan Huang, 1984). But in recent years, there has been improvement in the Von Neumann architecture by using the banks of memory that leads to parallel I/O to the memory. 

In general, parallelization can be achieved in two ways: vectorization within a single processor and under by using multiple processors.  The computer's speed is based on its ability to carry out floating-point operations, and computers are ranked according to benchmark performance of solving dense linear equations using subroutines from the LINPACK library (Jack  J  Dongarra et al., 1979).

![figure](https://drive.google.com/uc?export=view&id=1Q90h4Dg-tkcy2uGI1oO_JSI-5ug-Hu2P)
*Figure 1: (a) The Von Neumann architecure; (b) Early dual core Intel Xeon*


#### Multicore CPU
A multicore processor refers to two or more individual cores that are attached to the chip.   In recent years,  many chip producers have introduced multicore  CPUs, and this trend will increase gradually in the future, i.e., increasing the number of cores in a single chip.  Figure 1 (b) illustrates the 2 cores in a dual-core Intel Xeon processor from 2005.  In modern multicore CPUs, each core has its own L2 cache and shares the L3 cache among the cores.  The memory controller controls the memory banks in the Direct Random access memory  (DRAM).   Higher performance can be achieved using multithreading in multicore CPUs because single-thread performance is limited by the power wall, memory wall and IPL wall.


According to Flynn’s taxonomy, computer architectures can be classified into four categories based on how the instructions are executed on the data in the processor.  Single Instruction Single Data (SISD), which is quite simple and sequential. In Single Instruction Multiple Data (SIMD), the same instructions are executed on multiple data; this leads to data-level parallelism in the CPU. Figure 2 (b) illustrates the SIMD principle.  Multiple Instruction Single Data (MISD) is not very useful in reality, because a general program can not be easily mapped into this architecture.  In Multiple Instruction Multiple Data (MIMD), multiple instructions can be executed on multiple data in a single chip.  For example, in multicore CPUs, each core in the chip can do different tasks using multiple data,  which is achieved by thread-level parallelism.


![figure](https://drive.google.com/uc?export=view&id=15aeREvuOwv5huv-tG-EOFNQSNaktKLP-)
*Figure 2: (a) Memory hierarchy; (b) SIMD model*

In the memory hierarchy, the cache is small and can quickly be accessed by the
CPU. The cache holds the temporal information from the main memory, which might
be currently used by the processor. Cache, which is on-chip, is faster than the
off-chip memory. Figure 2 (a) shows the general memory hierarchy of the CPU.


#### Shared Memory Architectures

In shared-memory architectures, all the processors can exchange and access data from the global shared memory. Shared memory architectures are classified into three based on their memory access and bus network connection: Single shared memory model (Uniform Memory Access), sometimes called Symmetric Multi-Processing (SMP); Single shared memory with cache, and Distributed shared memory, this is called Non-Uniform Memory Access (NUMA). Figure 3 shows these architectures.



![figure](https://drive.google.com/uc?export=view&id=1bEvYJMOKZr68gfbt_YTUMO5pO1ycJCh8)
*Figure 3: Shared memory architectures*





## Day 4

### General Parallel Programming Concepts

This article shows the introduction to general parallel programming. Going through the section will help to see the difference between parallel programming on the CPU and GPUs (accelerators). 

### Shared Memory Programming-OpenMP:

OpenMP is based on thread-level parallelism, which can be created by the OpenMP programming model. OpenMP has three components, which enable parallel thread-level programming in the serial code. They are Compiler Directive, Runtime Library Routines, and Environmental Variables. 

In a simple way, it is called the fork and join method; that is, a single thread (a master thread) behaves sequentially, and when it enters the parallel region (FORK), it creates the slave threads (more parallel threads), which do the parallel computation. At the end of the parallel region (JOIN), slave threads will be disappeared (or destroyed), and the master thread will be continued until the end of the program. This can be seen in Figure 1, where the master thread creates slave threads, and after the parallel region, the slave threads are destroyed, and the master thread continues. 

![figure](https://drive.google.com/uc?export=view&id=1iMPC-A_wBCRbnz70eTDZzaWyTeaBgDB3)
*Figure 1: Fork-Join parallel approach (using the OpenMP programming model)*

Figure 2 shows the OpenMPs uniform memory access and non-uniform memory access model. In both cases, CPUs and memory are kept at the same compute node. 

![figure](https://drive.google.com/uc?export=view&id=1qtkpcs58aOEg0LC-rqxeCbjuTSbZ0P3g)
*Figure 2: Example of shared memory programming :(left) uniform memory access; (right)non-uniform memory access*

#### Distributed Memory Programming Model-Message Passing Interface (MPI):


## Day 5

### GPU Architecture

Here we study the generic GPU architecture, memory, cores and how the data communication has been improved between the GPUs over the years. 


* GPU has many cores: In general, GPUs has many cores compared to typical CPUs. But on the other hand, the frequency of the CPU is higher than the GPUs. That makes the CPU faster in computing compared to GPU, but GPUs outperform because it can handle many threads in parallel, which can process many data in parallel. 


![figure](https://drive.google.com/uc?export=view&id=1khagSDq91t7uZipGgPCM-Nf37yYFEraJ)
*Figure 1: CPU vs GPU archietecture* [source](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#introduction)


* In the GPU, cores are grouped into GPU Processing Clusters (GPCs), and each GPCs has its own Streaming Multiprocessors (SMs) and Texture Processor Clusters (TPCs). 

![figure](https://drive.google.com/uc?export=view&id=1RRDH7Wfaz8Vo3ueJl-mg1XW83Wd5QwaY)
*Figure 1: GPCs, SMs and TPCs* [source](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#introduction)


* Streaming multiprocessors in the GPU are capable of running multiple threads concurrently; this is called a single instruction multiple threads (SIMT)
architecture. SIMT enables programmers to achieve thread-level parallelism by using a single instruction with multiple threads that can handle multiple data. 

* Each streaming multiprocessor has its own instruction cache, registers, shared
memory, L1 cache, constant cache and texture cache; L2 cache, constant memory,
texture memory and global memory are shared among the multiple streaming multiprocessors.
    * **Global Memory** resides on the DRAM, which can read and write data from host to
device and device to host by using the CUDA API. The data in the global memory
12is accessible by all the threads until it is deallocated.
    * **Local Memory** is mainly used for the register spilling and holding the automatic
variables. Register spilling occurs when there is more register needed than is available. For both Fermi and Kepler, local memory is cached in L1 and L2 cache. Local memory is located in the off-chip memory of the GPU. 
    * **L1/Shared Memory** are on-chip memories. On Kepler, the main purpose of the L1
the cache is to hold stack data and spilt registers from the register memory and cache
the local data, whereas on Fermi, it caches global data as well as holds the spilt registers. Shared memory access leads to better coalesced memory access and reuse of the data again that is on-chip, which is as fast as the register if there is no memory bank conflict. Memory bank conflicts occur when two or more threads in the same warp try to access the same memory bank. The size of the L1 cache and shared memory can be modified by the programmer using the CUDA API.
    * **Constant Memory** is located in off-chip memory, which can read and write from/to host and device memory, but it is only readable from the threads. Cached constants are faster, and only a few kilobytes are available.
    * **Texture Memory** resides in off-chip memory like constant memory and is only readable from the threads, but it can read and write from/to host and device memory.
Texture cache is available as on-chip memory, which is slightly faster than the global
memory.

* SMs in the GPU are based on the scalable array multi-thread, which allows grid
and thread blocks of 1D, 2D, and 3D data. Therefore, programmers can write the
grid and block size to create a thread when executing the device kernel; this thread
block is called a cooperative thread array (CTA). GPU performance can be
improved if the “latency” is hidden; latency is the number of clock cycles needed
to execute the next warp in the SM. Each SM has single-precision CUDA cores,
double-precision units, special function units, and load/store units.

* A parallel execution is happening in the SMs and also in the“warps”.  One warp contains  32 threads;  warps can spawn across the SMs,  and each warp has its own instructions and registers. The multiprocessor occupancy is the ratio of active warps to the maximum number of warps supported on the GPU's multiprocessor. The Nvidia has a predetermined [CUDA Occupancy Calculator](https://docs.nvidia.com/cuda/cuda-occupancy-calculator/index.html) for a different Nvidia GPU architecture. 


### Nvidia Microarchitecture
* The Nvidia microarchitecture frequently releases the updated version of their GPUs microarchitecture. The following list shows some of their architectures:
Tesla (2006), Fermi (2010), Kepler (2012), Maxwell (2014), Pascal (2016), Volta (2017), Turing (2018), and Ampere (2020) 
* Each and every updated architecture have more advanced features compared to their previous releases. 
* There are different GPUs that are available based on each and every Nvidia microarchitecture.


### [Compute capabilities](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability)
* Nvidia GPUs features can be explored by programmers by using the compute capabilities. These compute capabilities determine the different functions that can be used by the programmer. Each and every Nvidia microarchitecture has its own compute capabilities, and usually, they are evoked in the compilation time as a compilation flag <code> -arch=compute_70</code>. For example, Tensor cores, half-precision, and mixed-precision functionality are achieved by specifying the compute capabilities. 


### [Unified Memory](https://developer.nvidia.com/blog/unified-memory-cuda-beginners/)
From Maxwell Nvidia, GPUs support a unified memory approach, which means both GPU and CPU can access, read, and write the same address with simplified memory allocation while you were writing the programme. 
![figure](https://drive.google.com/uc?export=view&id=1s3PbSwz5nRxSBh9xdEZ29bwSZQe7Pn2d)
*Figure 2: Unified memory* [source](https://developer.nvidia.com/blog/unified-memory-cuda-beginners/)

### [Nvidia NVLink](https://www.nvidia.com/en-us/design-visualization/nvlink-bridges/) 

NVLink is the latest technology from the Nvidia that connects the multiple GPU, which would transfer the data between the GPUs much faster compared to traditional PCIe-based solutions. Figure 3 shows how the GPUs are connected together using the [NVSwitch and NVLink](https://www.nvidia.com/en-us/data-center/nvlink/). And Figure 4 shows the data transfer performance in different Nvidia microarchitectures.     

![figure](https://drive.google.com/uc?export=view&id=1rVnHoiV_EMu_0ivrsEasc94UfeVgRXnb)
*Figure 3: Nvidia NVSwitch* [source](https://www.nvidia.com/en-us/data-center/nvlink/)

![figure](https://drive.google.com/uc?export=view&id=1IoVsFEDfP2kFaTOGvDe8flPncBmxPSOD) 
*Figure 4: Nvidia NVLink* [source](https://www.nvidia.com/en-us/data-center/nvlink/)


### To Know Hardware Specification of the GPU

Most of the time, when we run our GPU (CUDA) code, we do not know exactly what is the Nvidia GPU type we run. Understanding your GPU card will give know information that will be useful for writing optimized and efficient code. 

~~~bash
//-*-C++-*-
#include <stdio.h>
 
// Print device properties
void printDevProp(cudaDeviceProp devProp)
{
  printf("Major revision number:    %d\n",devProp.major);
  printf("Minor revision number:    %d\n",devProp.minor);
  printf("Name:                     %s\n",devProp.name);
  printf("Total global memory:      %u\n",devProp.totalGlobalMem);
  printf("Total shared memory per block: %u\n",devProp.
         sharedMemPerBlock);
  printf("Total registers per block:%d\n",devProp.regsPerBlock);
  printf("Warp size:                %d\n",devProp.warpSize);
  printf("Maximum memory pitch:     %u\n",devProp.memPitch);
  printf("Maximum threads per block:%d\n",devProp.maxThreadsPerBlock);
  for (int i = 0; i < 3; ++i)
    printf("Maximum dimension %d of block:%d\n", i,devProp.
           maxThreadsDim[i]);
  for (int i = 0; i < 3; ++i)
    printf("Maximum dimension %d of grid:%d\n",i,devProp.maxGridSize[i]);
  printf("Clock rate:               %d\n",  devProp.clockRate);
  printf("Total constant memory:    %u\n",  devProp.totalConstMem);
  printf("Texture alignment:        %u\n",  devProp.textureAlignment);
  printf("Concurrent copy and execution: %s\n",(devProp.deviceOverlap ?
                                                "Yes" : "No"));
  printf("Number of multiprocessors:%d\n",  devProp.multiProcessorCount);
  printf("Kernel execution timeout: %s\n",(devProp.kernelExec
                                       TimeoutEnabled ?"Yes" : "No"));
  return;
}

// Main program starts from here
int main()
{
  // Number of CUDA devices
  int devCount;
  cudaGetDeviceCount(&devCount);
  printf("CUDA Device Query...\n");
  printf("There are %d CUDA devices.\n", devCount);
 
  // Iterate through devices
  for (int i = 0; i < devCount; ++i)
    {
      // Get device properties
      printf("\nCUDA Device #%d\n", i);
      cudaDeviceProp devProp;
      cudaGetDeviceProperties(&devProp, i);
      printDevProp(devProp);
    }
 
  printf("\nPress any key to exit...");
  char c;
  scanf("%c", &c);
 
  return 0;
}
~~~

How to compile and execute:

~~~bash
$ Load the CUDA module (compiler)         
$ nvcc -arch=compute_70 -o device-query device-query.cu
$ ./device-query 
~~~

Output from the `quey-device`:

~~~bash 
// Example output
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

Similarly, using the `PGI` compiler will produce the GPU hardware specification. To see it, use:

~~~bash
pgaccelinfo
~~~







