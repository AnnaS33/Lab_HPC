
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cmath>
#include <time.h> 
#include <locale>
#define BLOCK_SIZE  16         

__global__ void matMultCuda(float* a, float* b, float* c, int n ) {
    
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	float sum = 0.0f;
	int ia = n*blockDim.y * blockIdx.y + n*threadIdx.y;
	int ib = blockDim.x * blockIdx.x + threadIdx.x;
	int ic = n * BLOCK_SIZE * blockIdx.y + BLOCK_SIZE * blockIdx.x;
	for ( int k = 0; k < n; k++ ) // calculating the element
		sum += a [ia + k] * b [ib + k*n];
	c [ic + n * ty + tx] = sum; 		
}

//filling with random numbers
void Filling_m(float *A,float *B, int r) {

	srand(time(NULL));
	for(int i=0;i<r; i++)
    		for (int j=0;j<r;j++){
        		A[i * r + j] =  rand()%100 ;
			B[i * r + j] =  rand()%100 ;
		}
}

 

//sequential multiplication of matrices
float matMulCPU(float* a, float* b, float* c, int N)
{
    clock_t start = clock();
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            c[i * N + j] = 0;
            for (int k = 0; k < N; ++k)
                c[i *N + j] += a[i *N + k] * b[k *N + j];
        }
    }

    clock_t end = clock();
    float ms = ((float)(end - start)/CLOCKS_PER_SEC)*1000;
    return ms;
}


//checking the equality of matrices
int equals_m(float* A, float* B, int n) {
	float e=0.000001;
   	for (int i = 0; i < n * n; i++) {
        	if (A[i] - B[i]>e) {
            	return 0;
        }
    }
    return 1;
}


int main(int argc, char* argv[])
{
	setlocale(LC_ALL, "Russian");
        int N = 512;       // matrix size is N*N
        printf("\n Matrix size: %u \n", N);

	float* a = new float[N * N];
        float* b = new float[N * N];
        float* C_cpu = new float[N * N];
        float* C_gpu = new float[N * N];

	//Fill in with random values
	Filling_m(a,b, N);
		
	//Matrices for CUDA
	float *d_A,*d_B,*d_C;
	cudaMalloc ( (void**)&d_A, N * N * sizeof ( float ) );
	cudaMalloc ( (void**)&d_B, N * N * sizeof ( float ) );
	cudaMalloc ( (void**)&d_C, N * N * sizeof ( float ) );	

	//Event Handler
	cudaEvent_t startt, stopp;
    	float gpuTime = 0.0f;
	cudaEventCreate(&startt);
    	cudaEventCreate(&stopp);

	//Тumber of threads and blocks
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    	dim3 blocks(N/threads.x, N/threads.y);
	
	//Mark the time
	cudaEventRecord(startt, 0);

	//Сopy the data to the device
	cudaMemcpy(d_A, a, N * N * sizeof(float), cudaMemcpyHostToDevice);
    	cudaMemcpy(d_B, b, N * N * sizeof(float), cudaMemcpyHostToDevice);
	
	//Core function
	matMultCuda<<< blocks, threads >>>(d_A,d_B,d_C,N);
	
	//Get the result
	cudaMemcpy(C_gpu, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

	//Stop point
   	cudaEventRecord(stopp, 0);
	cudaEventSynchronize(stopp);

	cudaEventElapsedTime(&gpuTime, startt, stopp);
	double ms = gpuTime;
    	printf("GPU time: %.2f ms\n", ms);

	float ms2=matMulCPU(a, b, C_cpu, N);
      	printf("CPU time: %.2f ms \n",ms2);

	bool yy=equals_m(C_cpu, C_gpu, N);
	printf("Statement: matrices are equal - %s \n", yy ? "true" : "false");
	
        cudaFreeHost( a);
       	cudaFreeHost (b);
        cudaFreeHost (C_cpu);
        cudaFreeHost (C_gpu);
	cudaEventDestroy(startt);
    	cudaEventDestroy(stopp);
    	cudaFree(d_A);
    	cudaFree(d_B);
    	cudaFree(d_C);

    return 0;
}

