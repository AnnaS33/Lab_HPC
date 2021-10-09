#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <locale>
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#define BLOCK_SIZE 256


__global__ void Summ(int *gpu_data, int *res,int N)
{	
	//выделение shared памяти
	__shared__ int sdata[BLOCK_SIZE];
	//номер нити
	unsigned int tid = threadIdx.x;
	
	//проходим по всему вектору получая частичные суммы
	int r = 0;
    	for (int i = tid; i < N; i += BLOCK_SIZE) {
        	r += gpu_data[i];
    	}

	sdata[tid] = r;
	__syncthreads();

	for(unsigned int j=1; j < blockDim.x; j *= 2) {
		if (tid % (2*j) == 0) {
			sdata[tid] += sdata[tid + j];
		}
	__syncthreads();
	}

	if (tid == 0){ 
		res[blockIdx.x] += sdata[0];
	}			
}

//сумма элементов вектора на CPU
float CPU_SumV(int * mass, int n){

	clock_t start = clock();

        int S=0;
	for(int i=0;i<n;i++){
		S+=mass[i];
	}
 	mass[0]=S;

	clock_t end = clock();
    	float ms = ((float)(end - start)/CLOCKS_PER_SEC)*1000;
    	return ms;

}

int main(int argc, char* argv[])
{
	srand(time(NULL));
	int n = 500000;
	int n2b = n * sizeof(int);
	
 	// Выделение памяти на хосте
	int * a=(int*)calloc(n,sizeof(int));
	int * b=(int*)calloc(n,sizeof(int));

	//заполнение случайными значениями
	for (int i=0 ; i<n ; ++i){
		a[i] = (int)rand()%100;
	}

	// Выделение памяти на устройстве
 	int* adev = NULL;
 	cudaMalloc((void**)&adev, n2b);
 	int* bdev = NULL;
 	cudaMalloc((void**)&bdev, n2b);

  // Создание обработчиков событий
 	cudaEvent_t start, stop;
 	float gpuTime = 0.0f;
 	cudaEventCreate(&start);
 	cudaEventCreate(&stop);

  // Копирование данных с хоста на девайс
 	cudaMemcpy(adev, a, n2b, cudaMemcpyHostToDevice);
	
  // Установка точки старта
 	cudaEventRecord(start, 0);

  //Запуск ядра
 	Summ<<< 1, BLOCK_SIZE >>>(adev, bdev,n);
	
 	// Копирование результата на хост
 	cudaMemcpy(b, bdev, n2b, cudaMemcpyDeviceToHost);
	
 	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&gpuTime, start, stop);
	double ms = gpuTime;

	printf("GPU time: %.9f ms\n", ms);
   	printf(" CPU time %.9f \n",CPU_SumV(a,n));
	printf(" Sum_gpu %i \n",b[0]);
	printf(" Sum_cpu %i \n",a[0]);
	 	
 	cudaEventDestroy(start);
 	cudaEventDestroy(stop);
 	cudaFree(adev);
 	cudaFree(bdev);
 	free(a);
 	free(b);
 
 	return 0;
}
