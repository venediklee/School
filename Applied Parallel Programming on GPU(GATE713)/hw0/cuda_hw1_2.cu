#include <stdio.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

/*
*	send matrices to gpu
*	sum each element with the other matrix's element that has the same position
*	store the result in the first matrix(same position)
*	get the first matrix from gpu
*	print the result
*
*	use PrintMatrixElement to print individual elements
*/

//kernel that executes on the CUDA device
__global__ void MatrixSummation(float *x, float*y, int maxIndex)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx<maxIndex)
	{
		x[idx] = x[idx] + y[idx];//save the result to the first(x) matrix to save space
	}
}

//used for error checking
void ErrorCheck(char *s)
{
	if (cudaGetLastError() != 0)//errorChecking
	{
		printf("error @ %s -> %s\n", s,cudaGetErrorString(cudaGetLastError()));
		//exit(-1);
	}
}

//used for printing matrices
void PrintMatrix(float *mtrx, int row, int column)
{

	printf("\n printing matrix of size %d by %d \n \n", row, column);
	for (int i = 0; i < row; i++)
	{
		printf("| ");
		for (int j = 0; j < column; j++)
		{
			printf("%f ", mtrx[i*column + j]);
		}
		printf("|\n\n");
	}
	printf("\n ######### \n printed matrix of size %d by %d \n ######### \n", row, column);
}

//prints the element in the (x,y) position of the matrix (does not do error checks)
// !! MY MATRIX STARTS AT (0,0) !!
// columnCount is the number of columns in matrix
void PrintMatrixElement(float *mtrx, int columnCount, int x, int y)
{
	printf("\n ######### \n printing matrix element at %d , %d \n \n", x, y);
	printf("%d", mtrx[columnCount*x + y]);
	printf("\n  \n printed matrix element at %d , %d \n ######### \n", x, y);
}

// main routine that executes on the host
int main(void)
{
	float *x_h, *x_d;//matrix1 pointers
	float *y_h, *y_d;//matrix2 pointers

	const int m = 1500;//m rows
	const int n = 10000;//n columns
	size_t matrixSize = m * n * sizeof(float);//size of memory space required for each matrix

	//allocate memory for CPU operations
	x_h = (float*)malloc(matrixSize);
	y_h = (float*)malloc(matrixSize);

	//set the arrays
	for (int i = 0; i<m; i++)//fill the first row first
	{
		for (int j = 0; j<n; j++)//i*n+j gives the appropriate position since 
								 //i*n is total # of elements in the earlier rows
								 //j is the index in that particular row
		{
			x_h[i*n + j] = i * n + j;//set values
			y_h[i*n + j] = i * n + j;//set values
		}
	}

	// ############### uncomment these for printing x and y matrices
	//PrintMatrix(x_h, m, n);
	//PrintMatrix(y_h, m, n);


	//allocate memory for GPU operations
	cudaMalloc((void**)&x_d, matrixSize);
	ErrorCheck("cudaMalloc(x)");

	cudaMalloc((void**)&y_d, matrixSize);
	ErrorCheck("cudaMalloc(y)");
	
	//copy the arrays from CPU to GPU
	cudaMemcpy(x_d, x_h, matrixSize, cudaMemcpyHostToDevice);
	ErrorCheck("cudaMemcpy(x)");

	cudaMemcpy(y_d, y_h, matrixSize, cudaMemcpyHostToDevice);
	ErrorCheck("cudaMemcpy(y)");

	//do calculation on GPU
	int blockSize = 32;
	int numberOfBlocks = (m*n) / blockSize + ((m*n) % blockSize == 0 ? 0 : 1);

	//benchmarking
	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	MatrixSummation << <numberOfBlocks, blockSize >> > (x_d, y_d, m*n);
	ErrorCheck("kernel call(MatrixSummation)");

	//copy x_d to x_h
	cudaMemcpy(x_h, x_d, matrixSize, cudaMemcpyDeviceToHost);
	ErrorCheck("cudaMemcpy(x 2)");

	//benchmarking
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float et;
	cudaEventElapsedTime(&et, start, stop);
	cudaEventDestroy(start); cudaEventDestroy(stop);
	//printf("\n @@@@@@@@@ benchmark result -> %f @@@@@@ \n", et);

	//print resulted matrix
	PrintMatrix(x_h, m, n);

	//free
	free(x_h); free(y_h);
	cudaFree(x_d); cudaFree(y_d);
	exit(0);




}
