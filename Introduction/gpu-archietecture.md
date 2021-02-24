### GPU architecture

* GPU has many cores: In general GPU has many cores compare to typical CPUs. But on the other hand frequency of the CPU is higher than the GPUs. That makes CPU faster in computing compare to GPU but still, GPUs win because, it can handle many threads, that can process many data in parallel. 


![figure](https://drive.google.com/uc?export=view&id=1khagSDq91t7uZipGgPCM-Nf37yYFEraJ)
*Figure 1: CPU vs GPU archietecture* [source](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#introduction)


* In the GPU, cores are grouped into GPU Processing Clusters (GPCs) and each GPCs has its own Streaming Multiprocessors (SMs) and Texture Processor Clusters (TPCs). 

![figure](https://drive.google.com/uc?export=view&id=1RRDH7Wfaz8Vo3ueJl-mg1XW83Wd5QwaY)
*Figure 1: GPCs, SMs and TPCs* [source](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#introduction)


* Streaming multiprocessors in the GPU are capable of running multiple threads concurrently; this is called a single instruction multiple thread (SIMT)
architecture. SIMT enables programmers to achieve thread level parallelism by using a single instruction with multiple threads that can handle multiple data. 

* Each streaming multiprocessor has their own instruction cache, registers, shared
memory, L1 cache, constant cache and texture cache; L2 cache, constant memory,
texture memory and global memory are shared among the multiple streaming multiprocessor.

* SMs in the GPU are based on the scalable array multi-thread, which allows grid
and thread blocks of 1D, 2D, and 3D data. Therefore, programmers can write the
grid and block size to create a thread when executing the device kernel; this thread
block is called a cooperative thread array (CTA). GPU performance can be
improved if the “latency” is hidden; latency is the number of clock cycles needed
to execute the next warp in the SM. Each SM has single-precision CUDA cores,
double-precision units, special function units, and load/store units.

* A parallel execution is happening in the SMs and also in the“warps”.  One warp contains  32 threads;  warps can spawn across the SMs,  and each warp has their own instructions and registers. The multiprocessor occupancy is the ratio of active warps to the maximum number of warps supported on a multiprocessor of the GPU. The Nvidia has a predetermined the [CUDA Occupancy Calculator](https://docs.nvidia.com/cuda/cuda-occupancy-calculator/index.html) different Nvidia GPU architecture. 


### Nvidia Microarchitecture
* The Nvidia microarchitecture frequently releases the updated version of their GPUs microarchitecture. The following list shows some of their architectures:
Tesla (2006), Fermi (2010), Kepler (2012), Maxwell (2014), Pascal (2016), Volta (2017), Turing (2018), and Ampere (2020) 
* Each and every updated architecture have more advanced features compared to their previous releases. 
* There are different GPUs that are available based on each and every Nvidia microarchitecture.


### [Compute capabilities](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability)
* Nvidia GPUs features can be explored by programmers by using the compute capabilities. These compute capabilities determine the different functions that can be used by the programmer. Each and every Nvidia microarchitecture has it's own compute capabilities, and usually, they are evoked in the compilation time as a compilation flag <code> -arch=compute_70</code>. For example, Tensor cores, half-precision, and mixed-precision functionality are achieved by specifying the compute capabilities. 


### [Unified Memory](https://developer.nvidia.com/blog/unified-memory-cuda-beginners/)
From Maxwell Nvidia, GPUs support a unified memory approach, which means both GPU and CPU can access read, and write the same address with simplified memory allocation while you writing the programme. 
![figure](https://drive.google.com/uc?export=view&id=1s3PbSwz5nRxSBh9xdEZ29bwSZQe7Pn2d)
*Figure 2: Unified memory* [source](https://developer.nvidia.com/blog/unified-memory-cuda-beginners/)

### [Nvidia NVLink](https://www.nvidia.com/en-us/design-visualization/nvlink-bridges/) 

NVLink is the latest technology from the Nvidia that connects the multiple GPU, which would transfer the data between the GPUs much faster compared to traditional PCIe-based solutions. Figure 3 shows how the GPUs are connected together using the [NVSwitch and NVLink](https://www.nvidia.com/en-us/data-center/nvlink/). And Figure 4 shows the data transfer performance in different Nvidia microarchitecture.     

![figure](https://drive.google.com/uc?export=view&id=1rVnHoiV_EMu_0ivrsEasc94UfeVgRXnb)
*Figure 3: Nvidia NVSwitch* [source](https://www.nvidia.com/en-us/data-center/nvlink/)

![figure](https://drive.google.com/uc?export=view&id=1IoVsFEDfP2kFaTOGvDe8flPncBmxPSOD)
*Figure 4: Nvidia NVLink* [source](https://www.nvidia.com/en-us/data-center/nvlink/)