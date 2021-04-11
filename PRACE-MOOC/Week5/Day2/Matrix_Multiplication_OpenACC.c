//-*-C++-*-
#include<stdio.h>
#include<stdlib.h>

void Matrix_Multiplication(float *a, float *b, float *restrict c, int width)   
{ 
  int length = width*width;
  float sum = 0;
#pragma acc kernels copyin(a[0:(length)], b[0:(length)]) copyout(c[0:(length)]) 
#pragma acc loop collapse(3) reduction (+:sum) 
  for(int row = 0; row < width ; ++row)                           
    {                                                             
      for(int col = 0; col < width ; ++col)                       
	{                                                         
	  for(int i = 0; i < width ; ++i)                         
	    {                                                     
	      sum += a[row*width+i] * b[i*width+col];      
	    }                                                     
	  c[row*width+col] = sum;
	  sum = 0;                            
	}
    }
}

int main()
{  
  printf("Programme assumes that matrix size is N*N \n");
  printf("Please enter the N size number \n");
  int N =0;
  scanf("%d", &N);

  // Initialize the memory on the host
  float *a, *b, *c;       
    
  // Allocate host memory
  a = (float*)malloc(sizeof(float) * (N*N));
  b = (float*)malloc(sizeof(float) * (N*N));
  c = (float*)malloc(sizeof(float) * (N*N));
  
  // Initialize host arrays
  for(int i = 0; i < (N*N); i++)
    {
      a[i] = 1.0f;
      b[i] = 2.0f;
    }
   
  // Device fuction call 
  Matrix_Multiplication(a, b, c, N);
  
  // Verification
  for(int i = 0; i < N; i++)
    {
      for(int j = 0; j < N; j++)
      	{
	  printf("%f ", c[j]);

	}
      printf("\n");
    }
  
  // Deallocate host memory
  free(a); 
  free(b); 
  free(c);

  return 0;
}
