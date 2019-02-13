#include <stdio.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>


/*
*	send matrix and vector to gpu
*	multiply each column of matrix with the vector(each matrix element with corresponding vector element)
*	store the result in the same matrix(same position)
*	get the matrix from gpu
*	sum each column of the matrix to a new vector
*	print the result
*
*	use PrintMatrixElement to print individual elements
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

//kernel that executes on CUDA device
__global__ void MatrixVectorMultiplication(float *mtrx, float *vect, int rows, int columns)
{
	//vect's length = rows
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int maxIndex = rows * columns;
	if (idx < maxIndex)
	{
		mtrx[idx] = mtrx[idx] * vect[idx/columns];//save the result to the mtrx for saving space
	}
}

//used for printing vectors
void PrintVector(float *vec,int length)
{
	printf("\n ######### \n printing vector of size %d \n \n", length);
	for (int i = 0; i < length; i++)
	{
		printf("%f  ", vec[i]);
	}
	printf("\n  \n printed vector of size %d \n ######### \n", length);
}

//prints the element in the (x,y) position of the matrix (does not do error checks)
// !! MY MATRIX STARTS AT (0,0) !!
// columnCount is the number of columns in matrix
void PrintMatrixElement(float *mtrx,int columnCount, int x, int y)
{
	printf("\n ######### \n printing matrix element at %d , %d \n \n", x, y);
	printf("%d",mtrx[columnCount*x+y]);
	printf("\n  \n printed matrix element at %d , %d \n ######### \n", x, y);
}

//used for printing matrices
void PrintMatrix(float *mtrx, int row, int column)
{

	printf("\n  ######### \n printing matrix of size %d by %d \n \n", row,column);
	for (int i = 0; i < row; i++)
	{
		printf("| ");
		for (int j = 0; j < column; j++)
		{
			printf("%f ", mtrx[i*column + j]);
		}
		printf("|\n\n");
	}
	printf("\n \n printed matrix of size %d by %d \n ######### \n", row,column);
}

int main(void)
{
	float *matr_d, *matr_h;//input matrix
	float *vec_d, *vec_h;//input vector   // I am assuming the size of the vector does not change in the code
	float *result_vector_h;//resulted vector

	const int m = 50000;//m rows && this is also the number of elements in the multiplying vector
	const int n = 700;//n columns && this is also the number of elements in the resulting vector

	//allocate memory & set values for matrix & vector on CPU
	matr_h = (float*)malloc(sizeof(float)*m*n);
	for (int i = 0; i < m; i++)//rows
	{
		for (int j = 0; j < n; j++)//columns
		{
			matr_h[i*n + j] = i * n + j;//set matrix values
		}
	}

	vec_h = (float*)malloc(sizeof(float)*m);
	for (int i = 0; i < m; i++)
	{
		vec_h[i] = i;//set vector values
	}

	//use these to print the input matrix and vector
	//PrintMatrix(matr_h, m, n);
	//PrintVector(vec_h, m);

	//allocate memory for matrix and vector on GPU
	cudaMalloc((void**)&matr_d, sizeof(float)*m*n);
	ErrorCheck("cudaMalloc (matr_d)");

	cudaMalloc((void**)&vec_d, sizeof(float)*m);
	ErrorCheck("cudaMalloc (vec_d)");

	//copy matr_h & vec_h to matr_d & vec_d
	cudaMemcpy(matr_d, matr_h, sizeof(float)*m*n, cudaMemcpyHostToDevice);
	ErrorCheck("cudaMemCpy (matr_d)");

	cudaMemcpy(vec_d, vec_h, sizeof(float)*m, cudaMemcpyHostToDevice);
	ErrorCheck("cudaMemcpy (vec_d)");

	int blockSize = 8;
	int numberOfBlocks = (m * n) / blockSize + ((m*n) % blockSize == 0 ? 0 : 1);


	// benchmarking
	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	MatrixVectorMultiplication << <numberOfBlocks, blockSize >> > (matr_d, vec_d, m, n);
	ErrorCheck("kernel call (matrixVectorMultiplication)");

	//allocate memory for result_h and copy the result from result_d
	//result_h = (float*)malloc(sizeof(float)*m*n);
	cudaMemcpy(matr_h, matr_d, sizeof(float)*m*n, cudaMemcpyDeviceToHost);
	ErrorCheck("cudaMemcpy (result_h)");

	// benchmarking
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float et;
	cudaEventElapsedTime(&et, start, stop);
	cudaEventDestroy(start); cudaEventDestroy(stop);
	//printf("\n @@@@@@@@@ benchmark result -> %f @@@@@@ \n", et);

	// use this to print the resulting matrix
	//PrintMatrix(matr_h, m, n);
	
	//set space for result_vector_h
	result_vector_h = (float*)malloc(sizeof(float)*n);
	//sum each column and set it to result_vector_h[columnIndex]
	for (int i = 0; i < n; i++)//columns
	{
		result_vector_h[i] = 0;//reset the value
		for (int j = 0; j < m; j++)//rows
		{
			result_vector_h[i] += matr_h[j*n + i];
		}
	}

	//print the resulting vector
	printf("resulting vector is -> \n | ");
	for (int i = 0; i < n; i++)
	{
		printf("%f  ", result_vector_h[i]);
	}
	printf("\n\n ############### \n\n");

	//cleanup
	free(matr_h); free(vec_h); free(result_vector_h);
	cudaFree(matr_d); cudaFree(vec_d);

	exit(0);
}



