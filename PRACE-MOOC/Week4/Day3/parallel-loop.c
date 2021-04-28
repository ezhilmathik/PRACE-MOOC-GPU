// Hello_World_OpenACC.c
#include<stdio.h>              
#include<openacc.h>

int main()
{ 
#pragma acc parallel loop
  for(int i = 0; i < 5; i++)
    {                                
      printf("Hello World!\n");
    }
  return 0;     
}
