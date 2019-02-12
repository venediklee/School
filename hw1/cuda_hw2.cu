#include <stdio.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
//!!!!!!!!!!!!!!!
//compile with nvcc -arch compute_11 cuda_hw2.cu


//m1 and m2 are the input matrices, m3 is resulting matrix, i1 and i2 are m1 and m2's rows, j1 and j2 are columns vice versa.(maxIndex is used to stop overcalculating in case we have an extra block)
__global__ void MatrixMultiplication(float *m1, float *m2, float *m3, int i1, int j1, int i2, int j2,int maxIndex)
{
	//get the thread id, get the float in m1[idx%(i1*j1)], get the float in m2[idx*j2+idx/i2],
		//multiply them and use atomicAdd at m3[idx/i2] with result
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	if (idx < maxIndex)
	{
		float a = m1[idx % j1 + j1*(idx/(i2*j2))];
		float b = m2[(idx%(i2*j2))/i2 + (idx%i2)*j2 ];
		float c = a * b;
		atomicAdd(&m3[idx / i2], c);
	}
}

//used for error checking
void ErrorCheck(char *c)
{
	if (cudaGetLastError() != 0)
	{
		printf("error @ %s -> %s \n", c, cudaGetErrorString(cudaGetLastError()));
		//exit(-1);
	}
}

//used for printing matrices
void PrintMatrix(float *mtrx, int row, int column)
{

	printf("\n  ######### \n printing matrix of size %d by %d \n \n", row, column);
	for (int i = 0; i < row; i++)
	{
		printf("| ");
		for (int j = 0; j < column; j++)
		{
			printf("%f ", mtrx[i*column + j]);
		}
		printf("|\n\n");
	}
	printf("\n \n printed matrix of size %d by %d \n ######### \n", row, column);
}

int main(void)
{
	//setting matrix sizes
	const int i1 = 5;//first and resulting matrices have i1 rows
	const int j1 = 2;//first matrix has j1 columns
	const int i2 = j1;//second matrix has i2=j1 rows
	const int j2 = 2;//second and resulting matrices have j2 columns

	//2 matrix pointers(on host)
	float *m1_h, *m2_h;

	//allocate memory for matrices(on host)
	m1_h = (float*)malloc(sizeof(float)*i1*j1);
	m2_h = (float*)malloc(sizeof(float)*i2*j2);
	
	//set values to matrices(on host)
	for (int i = 0; i < i1*j1; i++) m1_h[i] = i;
	for (int i = 0; i <i2*j2; i++) m2_h[i] = i;

	//calculate(?) blockSize and number of blocks for kernel
	int numberOfElements = i2 * j2 * i1;//this is the total number of multiplications required, in other words total number of threads
	int blockSize = 32*4;//my device has 128 threads per SM
	int numberOfBlocks = numberOfElements / blockSize + (numberOfElements % blockSize == 0 ? 0 : 1);

	//create 3 matrix pointers, last one being the resulting matrix (on device)
	float *m1_d, *m2_d, *m3_d;

	//allocate memory on device
		// DO NOT READ THIS COMMENT -- OLD //allocate the second matrix first in order to prevent the device from deallocating and reallocating the first matrix --which is not pinned-- on the hard drive
	cudaMalloc((void**)&m2_d, sizeof(float)*i2*j2); ErrorCheck("cudaMallocHost(m2_d)");//allocate second matrix's memory as pinned since we will be using it more often
	cudaMalloc((void**)&m1_d, sizeof(float)*i1*j1); ErrorCheck("cudaMalloc(m1_d)");
	cudaMalloc((void**)&m3_d, sizeof(float)*i1*j2); ErrorCheck("cudaMalloc(m3_d)");

	//set m3_d to 0's (resulting matrix) since I'll be using atomicAdd
	cudaMemset(m3_d, 0, sizeof(float)*i1*j2); ErrorCheck("cudaMemset(m3_d)");

	//copy matrices to device
	cudaMemcpy(m2_d, m2_h, sizeof(float)*i2*j2, cudaMemcpyHostToDevice); ErrorCheck("cudaMemcpy(m2_d)");
	cudaMemcpy(m1_d, m1_h, sizeof(float)*i1*j1, cudaMemcpyHostToDevice); ErrorCheck("cudaMemcpy(m1_d)");
	
	//activate kernel
		//send each matrices pointers(on device) and their respective row and column size
	MatrixMultiplication << <numberOfBlocks, blockSize >> > (m1_d, m2_d, m3_d, i1, j1, i2, j2, numberOfElements); ErrorCheck("kernel call(MatrixMultiplication)");

	//copy the result back to host
	float *m3_h;
	m3_h = (float*)malloc(sizeof(float)*i1*j2);
	cudaMemcpy(m3_h, m3_d, sizeof(float)*i1*j2, cudaMemcpyDeviceToHost); ErrorCheck("cudaMemcpy(m3_h)");

	//print the result
	PrintMatrix(m3_h, i1, j2);

	//free
	free(m1_h); 
	free(m2_h); 
	free(m3_h);
	cudaFree(m1_d); cudaFree(m2_d); cudaFree(m3_d);

}
