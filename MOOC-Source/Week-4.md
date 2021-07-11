## Week 4: OpenACC (basic): Introduction to OpenACC Programming

In this week, we see the introduction to OpenACC, in particular, how the OpenACC directives are defined, where and how it can be called in the C/C++ or Fortran programming application.

## Day 1

## Introduction to OpenACC

In this article, we show and discuss the introduction to OpenACC. We consider both C/C++ and Fortran programming language here.


In this article, we will study the basics of OpenACC programing structure and compiler options flag options for different programing language. 

OpenACC is a user-driven directive-based performance-portable parallel programming model. It is designed for scientists and engineers interested in porting their codes to a wide variety of heterogeneous HPC hardware platforms and architectures with significantly less programming effort than required with a low-level model. The OpenACC specification supports C, C++, Fortran programming languages and multiple hardware architectures, including X86 & POWER CPUs, NVIDIA GPUs, and Xeon KNL. The below table provides the compiler command for the different programming languages. 

| *Compiler or Tool* | *Language or Function* | *Command* |
|-|-|-|
|__PGF77__|ANSI FORTRAN 77|`pgf77`|
|__PGFORTRAN__|ISO/ANSI Fortran 2003|`pgfortran`|
|__PGCC__|ISO/ANSI C11 and K&R C|`pgcc`|
|__PGC++__|ISO/ANSI C++14 with GNU compatibility|`pgc++` on Linux and macOS|
|__PGI Debugger__|Source code debugger|`pgdbg`|
|__PGI Profiler__|Performance profiler|`pgprof`|

*Table 1: PGI compiler command options and different programming languages*

The below listings show the basic syntax in OpenACC for `C/C++` and `Fortran`. To be able to use the OpenACC API in the application, please use: `#include "openacc.h"` for `C/C++` and `use openacc` for `Fortran`.

~~~openmp
// C/C++
#include "openacc.h"
#pragma acc <directive> [clauses [[,] clause] . . .] new-line
<code>
~~~

~~~openmp
// Fortran
use openacc
!$acc <directive> [clauses [[,] clause] . . .]
<code>
~~~

The following points explain the basic syntax entries in the above listing. 

* A __`pragma`__ instructs the compiler to run the region in parallel by a team of threads. 
* __`acc`__ instructs the compiler to use the OpenACC directive definitions. 
* A __`directive`__ is an __instruction__ to the compiler on how the parallel region's code block can be executed. In OpenACC, three directives are available: *Compute directives*, *Data management directives*, and *Synchronization directives*. 
   * __Compute directives__: This will enable the data parallelization with multiple threads. The OpenACC directives are: `parallel`, `kernels`, `routine`, and `loop`
   * __Data management directives__: These directives help reduce unnecessary data movement between the different memories. By default, compiler directive does this data movement, but they are not well optimised. The OpenACC data directives are: `data`, `update`, `cache`, `atomic`, `declare`, `enter data`, and `exit data`. 
   * __Synchronization directives__: To support the task parallelization, OpenACC allows multiple constructs to be executed currently. In some situation, these tasks should be controlled; to do that, OpenACC provides a `wait` directive. 
* A __`clause`__ is an argument to the directives, which instructs the compiler with more information on how to behave the directives. Three clauses exist in OpenACC: *Data handling*, *Work distribution* and *Control flow*.
    * __Data handling__: These clauses override the compiler analysis for the variables. A few examples of clauses are: `default`, `private`, `firstprivate`, `copy`, `copyin`, `copyout`, `create`, `delete`, and `deviceptr`.
    * __Work distribution__: It will able programmers to control the threads in the parallel region. A few available work distribution clauses are: `seq`, `auto`, `gang`,
`worker`, `vector`, `tile`, `num_gangs`, `num_workers`, and `vector_length`.
    * __Control flow__: It instructs the compiler to control the parallel directives' execution. For example, `if` or `if_present`, `independent`, `reduction`, `async` and `wait`.


There are other compilers also exciting for the OpenACC model, the Table 2 shows different compiler flags for and additional flags for the various compiler options.

|*Compiler*			|*Compiler	Flags*	|*Additional Flags*|
|--|--|--|
|__PGI__|-acc|-ta=target	architecture -Minfo=accel	|
|__GCC__	|-fopenacc	|	–foffload=offload	target	|
|__OpenUH__	|Compile:	-fopenacc	Link: -lopenacc |-Wb,	-accarch:target	architecture	|
|__Cray__| C/C++: -h pragma=acc Fortran: -h acc,noomp| -h msgs|

*Table 2: Various compiler and its compiler flags*


> In the next weeks, we will learn the programming techniques in the OpenACC. Since OpenACC is the directive-based model, so we will study the theory and examples based on the directives. 


## Day 2
## Functionality of OpenACC

In this article, we show and discuss an overview of the functionality of the OpenACC.

In this article, we will see a few of the important functionality of the OpenACC. 

The OpenACC paradigm can be defined into three categories, and they are *Incremental*, *Single source*, and *Low learning curve*. The following section describes those categories. Figure 1 shows the OpenACC implementation cycle, where a serial code can be converted to better parallel code, 

![figure](https://drive.google.com/uc?export=view&id=19cvf80jYDIbztE_MemiHlvfLyvYHKd8d)
*Figure 1: OpenACC implementation cycle*

* Incremental  
    * An original code that is exiting code (sequential or parallel) can be maintained.
    * Easily add the annotation and make a particular region to be executed in parallel.
    * With an initial annotation, check the correctness, later add more wherever it is needed.

          // Serial code (SAXPY call)
          for (int i=0; I <N; i++)
          {
          y[i] = a*x[i] + y[i];
          }

          // OpenACC (SAXPY call)
          #pragma acc parallel loop
          for (int i=0; I < N; i++)
          {
          y[i] = a*x[i] + y[i];
          }
        
* Single source 
   * Same code can be used on multiple architectures.
   * A palletization strategy is determined by the compiler. 
   * Sequential code can still be maintained.

          int main()
          {
          .
          // OpenACC (SAXPY call)
          #pragma acc parallel loop
          for(int i=0; I < N; i++)
              {
              y[i] = a*x[i] + y[i];
              }
          }

* Low learning curve
   * OpenACC is easy to learn, use and maintain.
   * No need to learn the low-level programming details for different parallel architecture.

          int main()
          {
          <sequential code>
          #pragma acc kernels // compiler hint
              {
              <parallel code>
              } 
          }



## OpenACC Compute Constructs

In this section, we show and discuss the introduction to OpenACC compute constructs. These are essential to learn how to parallelize the serial code.

The OpenACC has four compute constructs; they are `parallel`, `kernels`, `loop`, and `routine`, which distribute the work among the parallel threads in a particular region where OpenACC constructs are defined. We will see now how to use them in the OpenACC application for both `C/C++` and `Fortran`. 

#### `parallel` Construct 

For C/C++:

~~~openmp
#pragma acc parallel 
  for(int i = 0; i < 5; i++)
    {                                
      printf("Hello World!\n");
    }
~~~

For Fortran:

~~~fortran 
  !$acc parallel
  do i = 1, 5
     print *, "hello world"
  end do
  !$acc end parallel
~~~


#### `kenrels` Construct 

For C/C++:

~~~openmp
#pragma acc kernels
  for(int i = 0; i < 5; i++)
    {                                
      printf("Hello World!\n");
    }
~~~

For Fortran:

~~~fortran 
  !$acc kernels
  do i = 1, 5
     print *, "hello world"
  end do
  !$acc end kernels
~~~


### Difference between `parallel` and `kernels`

Even though both `parallel` and `kernels` compute constructs, yet, there is an important difference between them. There is two difference are exiting between them, they are:

 
#### `loop` Construct 

For C/C++:

~~~openmp
#pragma acc parallel loop
  for(int i = 0; i < 5; i++)
    {                                
      printf("Hello World!\n");
    }
//-----------------------//
#pragma acc kernels loop
  for(int i = 0; i < 5; i++)
    {                                
      printf("Hello World!\n");
    }
~~~

For Fortran:

~~~fortran
  !$acc parallel loop
  do i = 1, 5
     print *, "hello world"
  end do
  !$acc end parallel
!!-----------------------!!
  !$acc kernels loop
  do i = 1, 5
     print *, "hello world"
  end do
  !$acc end kernels
~~~


#### `routine` Contruct 

For C/C++:

~~~openmp
#pragma acc routine seq
extern int simplecompute(int *a)

#pragma acc routine seq
int simplecompute(int *a)
{
     return a%2;
}

void maincompute(int *a, int N)
{
   #pragma acc parallel loop
    for (i=0; I < N; i++)
        x[i] = simplecompute(i)
}
~~~

For Fortran:

~~~fortran 
!$acc routine(primary_func) gang

interface 
  subroutine primary_func(a,b,x,n)
   !$acc routine gang
   real a(*), b(*)
   real, value :: x
   integer, value :: n
  end subroutine
end interface

use secondary_func

!$accparallel primary_func(a,b,x) num_gangs(n/32) vector_length(32)
  call secondary_func(a, b, x, n)
!$acc end parallel
~~~



## Day 4
## The Data Environment in OpenACC

In this section, we show a few of the data handling clauses that are available in OpenACC. These are very important APIs for data transfer and writing.

## The Data Environment in OpenACC

### Data Cluases in OpenACC

OpenACC directives have clauses that can be explicitly used to transfer the data between CPU (host) to GPUs(device). OpenACC also provides the `cache` clause; this can be used for frequently used variables in the parallel loop directive. In this section, we show a few examples of using some of the data calluses in OpenACC. 

The data clauses that are available in OpenACC are explained as follows:

* `copy(list)`: Allocates memory on GPU and copies data from the host to GPU when entering region and copies data to the host when exiting the region, see example [copy.c](https://raw.githubusercontent.com/ezhilmathik/PRACE-MOOC-GPU/main/PRACE-MOOC/Week4/Day4/copy.c?token=AK3BVJSNC6OYD6BMDNXBRK3APTSEQ).

      void Vector_Addition(float *a, float *b, float *restrict c, int n) 
      {
      #pragma acc data copyin(a[0:n], b[0:n]) copy(c[0:n])
        {
      #pragma acc kernels loop //copyin(a[:n], b[0:n]) copy(c[0:n])
          for(int i = 0; i < n; i ++)
            {
	      c[i] = c[i] + a[i] + b[i];
            }
        }
      }


* `copyin(list)`: Allocates memory on GPU and copies data from the host to GPU when entering the region; see example [copyin.c](https://raw.githubusercontent.com/ezhilmathik/PRACE-MOOC-GPU/main/PRACE-MOOC/Week4/Day4/copyin.c?token=AK3BVJQFKI6SLG7DW4B4HDDAPTSG4).

      void Vector_Addition(float *a, float *b, float *restrict c, int n) 
      {
      #pragma acc data copyin(a[0:n], b[0:n]) copyout(c[0:n])
        {
      #pragma acc kernels loop //copyin(a[:n], b[0:n]) copyout(c[0:n])
          for(int i = 0; i < n; i ++)
            {
	      c[i] = a[i] + b[i];
            }
        }
      }

* `copyout(list)`: Allocates memory on GPU and copies data to the host when exiting the region; see example [copyout.c](https://raw.githubusercontent.com/ezhilmathik/PRACE-MOOC-GPU/main/PRACE-MOOC/Week4/Day4/copyout.c?token=AK3BVJRTKWT6IAHBQ6ZP2WTAPTSI6).

      void Vector_Addition(float *a, float *b, float *restrict c, int n) 
      {
      #pragma acc data copyin(a[0:n], b[0:n]) copyout(c[0:n])
        {
      #pragma acc kernels loop //copyin(a[:n], b[0:n]) copyout(c[0:n])
          for(int i = 0; i < n; i ++)
            {
	      c[i] = a[i] + b[i];
            }
        }
      }

* `create(list)`: Allocates memory on GPU but does not copy, see examples [create-1.c](https://raw.githubusercontent.com/ezhilmathik/PRACE-MOOC-GPU/main/PRACE-MOOC/Week4/Day4/create-1.c?token=AK3BVJU2VJSULR7J27ZH4WTAPTSMG), [create-2.c](https://raw.githubusercontent.com/ezhilmathik/PRACE-MOOC-GPU/main/PRACE-MOOC/Week4/Day4/create-2.c?token=AK3BVJQJIWW6P4CGYZRA4ALAPTSNM) and [create-3.c](https://raw.githubusercontent.com/ezhilmathik/PRACE-MOOC-GPU/main/PRACE-MOOC/Week4/Day4/create-3.c?token=AK3BVJT5FYMOOY5BO754B5DAPVTKK).

      void Vector_Addition(float *a, float *b, float *restrict c, int n) 
      {
      #pragma acc data copyin(a[0:n], b[0:n]) create(c[0:n]) copyout(c[0:n]) 
        {
      #pragma acc kernels loop //copyin(a[:n], b[0:n]) create(c[0:n]) copyout(c[0:n])
          for(int i = 0; i < n; i ++)
            {
	      c[i] = a[i] + b[i];
            }
        }
      }


* `present(list)`: Data is already present on GPU from another containing data region.
* `deviceptr(list)`: The variable is a device pointer (e.g., CUDA) and can be used directly on the device.


#### `copyin` and `copyout` (example for vection addition)

For C/C++:

~~~oprnmp
#pragma acc kernels loop copyin(a[0:n], b[0:n]), copyout(c[0:n])
for(int i = 0; I < n; i++)
{
    c[i] = a[i] + b[i];
}
~~~

For Fortran:

~~~fortran
!$acc kernels loop copyin(a(1:n), b(1:n)), copyout(c(1:n))
DO i = 1, N
    c(i) = a(i) + b(i)
END DO
!$acc end kernels
~~~

#### `cache` Clause

For C/C++:

~~~openmp
#pragma acc parallel loop 
for (int i = 0; i < N; i++)
{
#pragma acc cache (a[j])
    a[i] = a[i] * 2.0
}
~~~

For Fortran:

~~~fortran
!$acc parallel loop
DO i = 0, N
!$acc cache (a(i))
    a(i) = a(i) * 2.0   
END DO
!$acc end parallel 
~~~



~~~openacc
/// Unstructured 
//Can have multiple starting/ending points
//Can branch across multiple functions
//Memory exists until explicitly deallocated
#pragma acc enter data copyin(a[0:N],b[0:N]) create(c[0:N])
#pragma acc parallel loop 
for(inti = 0; i < N; i++)
{
c[i] = a[i] +b[i];
}
#pragma acc exit data copyout(c[0:N]) delete(a,b)
~~~



~~~openacc
///structured
//Must have explicit start/end points
//Must be within a single function
//Memory only exists within the data region
#pragma acc data copyin(a[0:N],b[0:N]) copyout(c[0:N])
{
#pragma acc parallel loop 
for(inti = 0; i < N; i++)
{
c[i] = a[i] +b[i];
}
}
~~~

## Day 5

## Programming in OpenACC

In this section, we show a gentle introduction to OpenACC programming model for C/C++ and Fortran programming languages.


## C/C++ and Fortran programing model in OpenACC

In this article, we see how to print out the hello world problem using the OpenACC for C/C++ and Fortran programming languages. 

####  C/C++:

The below coding example shows the [Hello_World.c](https://github.com/ezhilmathik/PRACE-MOOC-GPU/blob/main/PRACE-MOOC/Week4/Day5/Hello_World.c) in C/C++ programming language. 

~~~ cpp
// Hello_World.c           
void Print_Hello_World()    
{   
  for(int i = 0; i < 5; i++)
    {                                
      printf("Hello World!\n");
    }
} 

##### Compilation #####
pgcc -fast -Minfo=all -ta=tesla -acc Hello_World.c

##### Compilation output #####
Print_Hello_World:
      6, Loop not vectorized/parallelized: contains call
main:
     14, Print_Hello_World inlined, size=5 (inline) file Hello_World.c (4)
           6, Loop not vectorized/parallelized: contains call
~~~

As we can see from the above example compilation output, it clearly shows that the `loop` at line 6 (from the Print_Hello_World function) is not parallelized. Now, we will use the OpenACC compute directive to parallelize the `for` loop in the function, [Hello_World_OpenACC.c](https://github.com/ezhilmathik/PRACE-MOOC-GPU/blob/main/PRACE-MOOC/Week4/Day5/Hello_World_OpenACC.c) can be seen in the below example. 

~~~cpp
// Hello_World_OpenACC.c
void Print_Hello_World()    
{
#pragma acc kernels
  for(int i = 0; i < 5; i++)
    {                                
      printf("Hello World!\n");
    }
} 

##### Compilation #####
pgcc -fast -Minfo=all -ta=tesla -acc Hello_World_OpenACC.c

##### Compilation output #####
Print_Hello_World:
      8, Loop is parallelizable
         Generating Tesla code
          8, #pragma acc loop gang, vector(32) /* blockIdx.x threadIdx.x */
~~~

We can see that `for loop` in the Print_Hello_World() function is parallelized; this can be seen in the compilation output. As we can notice here, the `for loop` is going to be executed with 32 threads. This is automatically chosen/generated by the compiler. Depends on the `N` number of the iterations, an optimized number of threads will be chosen by the compiler using the combinations of `gang`, `worker`, and `vector`. This will be similar to in CUDA, such as blocks, warps, and threads. 

> Here, we have used the `kernels` compute directive to parallelize the `for` loop. However, to effectively parallelize the loop, we should use the `loop` clause along with the `kernels` directive. This is explained in the OpenACC compute construct section. 

#### For Fortan 

The below coding example shows the serial Fortran [Hello_World.f90](https://github.com/ezhilmathik/PRACE-MOOC-GPU/blob/main/PRACE-MOOC/Week4/Day5/Hello_World.f90). Here, the compiler tells us that the `for loop` is not parallelized at the subroutine Print_Hello_World(); this can be noticed from the compilation output. 

~~~fortran 
!!! Hello_World.f90
subroutine Print_Hello_World()
  integer :: i
  do i = 1, 5
     print *, "hello world"
  end do
end subroutine Print_Hello_World

##### Compilation #####
pgfortran -fast -Minfo=all -ta=tesla -acc Hello_World.f90

##### Compilation output #####
print_hello_world:
      3, Loop not vectorized/parallelized: contains call
~~~

The parallel version of [Hello_World_OpenACC.f90](https://github.com/ezhilmathik/PRACE-MOOC-GPU/blob/main/PRACE-MOOC/Week4/Day5/Hello_World_OpenACC.f90) can be seen in the below example. Here, the loop is parallelized by using `kernels`; this can be already seen in the compilation output. 

~~~fortran 
!!! Hello_World_OpenACC.f90
subroutine Print_Hello_World()
  integer :: i
  !$acc kernels
  do i = 1, 5
     print *, "hello world"
  end do
  !$acc end kernels
end subroutine Print_Hello_World

##### Compilation #####
pgfortran -fast -Minfo=all -ta=tesla -acc Hello_World_OpenACC.f90

##### Compilation output #####
print_hello_world:
      4, Loop is parallelizable
         Generating Tesla code
          4, !$acc loop gang, vector(32) ! blockidx%x threadidx%x
~~~


