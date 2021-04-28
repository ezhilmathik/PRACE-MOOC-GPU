//-*-C++-*-
#include<stdio.h>
#include<stdlib.h>

float * Matrix_Multiplication(float *h_a, float *h_b, float *h_c, int width)   
{ 

#pragma acc parallel copyin(h_a[0:(width*width)],h_b[0:(width*width)]), copyout(h_c[0:(width*width)])                                                                
  for(int row = 0; row < width ; ++row)                           
    {                                                             
      for(int col = 0; col < width ; ++col)                       
        {                                                         
          float single_entry = 0;                                       
          for(int i = 0; i < width ; ++i)                         
            {                                                     
              single_entry += h_a[row*width+i] * h_b[i*width+col];      
            }                                                     
          h_c[row*width+col] = single_entry;                            
        }                                                         
    }   
  return h_c;           
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
