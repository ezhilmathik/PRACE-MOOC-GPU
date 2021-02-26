//-*-C++-*-
#include<iostream>

using namespace std;


float * matrix_mul(float *h_a, float *h_b, float *h_c, int width)   
{                                                                 
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
  
  cout << "Programme assumes that matrix size is N*N "<<endl;
  cout << "Please enter the N size number "<< endl;
  int N=0;
  cin >> N;

  // Initialize the memory on the host
  float *a, *b, *c;       
    
  // Allocate host memory
  a   = (float*)malloc(sizeof(float) * (N*N));
  b   = (float*)malloc(sizeof(float) * (N*N));
  c   = (float*)malloc(sizeof(float) * (N*N));
  
  // Initialize host arrays
  for(int i = 0; i < (N*N); i++)
    {
      a[i] = 1.0f;
      b[i] = 2.0f;
    }
   
  // Device fuction call 
  matrix_mul(a, b, c, N);

  // Verification
  for(int i = 0; i < N; i++)
    {
      for(int j = 0; j < N; j++)
      	{
	  cout << c[j] <<" ";
	}
      cout << " " <<endl;
    }
    
  // Deallocate host memory
  free(a); 
  free(b); 
  free(c);

  return 0;
}
