#include <stdio.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/*
*	send arrays to gpu
*	multiply each element with the other array's element that has the same position
*	store the result in the first array(same position)
*	get the first array from gpu
*	print the result
*/

//used for error checking
void ErrorCheck(char *c)
{
	if (cudaGetLastError() != 0)
	{
		printf("error @ %s -> %s \n", c, cudaGetErrorString(cudaGetLastError()));
		//exit(-1);
	}
}

// Kernel that executes on the CUDA device
__global__ void ArrayMultiplication(float *x, float*y, int maxIndex)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx<maxIndex)
	{
		x[idx] = x[idx] * y[idx];//store the result in x to use less memory
	}
}

// main routine that executes on the host
int main(void)
{
	float *x_h, *x_d;//array1 pointers
	float *y_h, *y_d;//array2 pointers
	const int numElements = 8000000;//# of elements in arrays

	size_t size = numElements * sizeof(float);//size of memory space required for each array

	//allocate memory for CPU operations
	x_h = (float*)malloc(size);
	y_h = (float*)malloc(size);

	//set the arrays
	for (int i = 0; i<numElements; i++)
	{
		x_h[i] = (float)i;
		y_h[i] = (float)i;
	}

	//allocate memory for GPU operations
	cudaMalloc((void**)&x_d, size);
	ErrorCheck("cudaMalloc(x)");

	cudaMalloc((void**)&y_d, size);
	ErrorCheck("cudaMalloc(y)");

	//send the arrays on CPU to GPU
	cudaMemcpy(x_d, x_h, size, cudaMemcpyHostToDevice);
	ErrorCheck("cudaMemcpy(x)");

	cudaMemcpy(y_d, y_h, size, cudaMemcpyHostToDevice);
	ErrorCheck("cudaMemcpy(y)");

	//do calculation on GPU
	int blockSize = 8;//# of threads in each block
	int numberOfBlocks = numElements / blockSize + (numElements % blockSize == 0 ? 0 : 1);

	//benchmarking
	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	ArrayMultiplication <<< numberOfBlocks, blockSize >>>(x_d, y_d, numElements);
	ErrorCheck("kernelCall(ArrayMultiplication)");
	
	//copy the x_d(this is where the result of multiplication is stored) to x_h
	cudaMemcpy(x_h, x_d, size, cudaMemcpyDeviceToHost);
	ErrorCheck("cudaMemcpy(x 2)");
	
	//benchmarking
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float et;
	cudaEventElapsedTime(&et, start, stop);
	cudaEventDestroy(start); cudaEventDestroy(stop);
	//printf("\n @@@@@@@@@ benchmark result -> %f @@@@@@ \n", et);

	//sum the contents of x_h
	float result = 0;
	for (int i = 0; i<numElements; i++)
	{
		result += x_h[i];
	}
	printf("%f\n", result);

	//free
	free(x_h); free(y_h);
	cudaFree(x_d); cudaFree(y_h);

	exit(0);

}
