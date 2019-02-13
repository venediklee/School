#include "kernels.cuh"


//KERNELS


//used for error checking
void ErrorCheck(char *c)
{
	if (cudaGetLastError() != 0)
	{
		printf("error @ %s -> %s \n", c, cudaGetErrorString(cudaGetLastError()));
		//exit(-1);
	}
}


//TODO PARALLEL REDUCTION
//img is pointer to image
//width & height are dimensions of image
//step is the step(or pitch) value of the Source image
//paralel reduces the min&max values to img[0] & img[131072] respectively
//each thread is responsible for 4 index'
//call this function with 512*512/4=65536 threads
__global__ void kernel1(Npp8u *img, int width, int height, int step)
{
	// cant use shared memory(max shared data 0xc000)(my shared data 0x40000)
	//__shared__ Npp8u sdata[262144];//first half is used for finding min, second half is used for finding max

	unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;

	Npp8u temp;

	//indexs are used for getting the correct position on memory
	unsigned int firstIndex = (tid / width)*step + tid % width;//no need to calculate at each step
	unsigned int secondIndex = (65536 / width)*step + 65536 % width + firstIndex;
	unsigned int thirdIndex = (65536 / width)*step + 65536 % width + secondIndex; //no need to calculate at each step
	unsigned int fourthIndex = (65536 / width)*step + 65536 % width + thirdIndex;

	//first compare each element and store the bigger ones to latter part of the image
	if (img[firstIndex]>img[thirdIndex])//branching
	{
		temp = img[firstIndex];
		img[firstIndex] = img[thirdIndex];
		img[thirdIndex] = temp;
	}
	//no need to check else since img[firstIndex] is already img[firstIndex]

	if (img[secondIndex] > img[fourthIndex])//branching
	{
		temp = img[secondIndex];
		img[secondIndex] = img[fourthIndex];
		img[fourthIndex] = temp;
	}
	//no need to check else since img[thirdIndex] is already img[thirdIndex]
	__syncthreads();

	//compare firstIndex with secondIndex etc.
	//put the min to img[firstIndex], max to img[thirdIndex]
	for (unsigned int i = 65536; i>0; i=i / 2)//each thread checks min and max arrays(1 array in reality)
	{
		//only i/65536 of threads are active after this point(warps are fully utilized)
		if (tid < i)
		{
			
			secondIndex = (i / width)*step + i % width + firstIndex;//can be optimized by compiler
			fourthIndex = (i / width)*step + i % width + thirdIndex;//can be optimized by compiler

			//finding min. operations // use first&secondIndex
			if (img[firstIndex]>img[secondIndex])//branching
			{
				img[firstIndex] = img[secondIndex];
			}
			//no need to check else since img[firstIndex] is already img[firstIndex]

			//finding max. operations // use third&fourthIndex
			if (img[thirdIndex] < img[fourthIndex])//branching
			{
				img[thirdIndex] = img[fourthIndex];
			}
			//no need to check else since img[thirdIndex] is already img[thirdIndex]


		}
		__syncthreads();//finish the iteration step
	}
	__syncthreads();//finish for loop
}


//pSrc_Dev is pointer to image
//nWidth & nHeight are dimensions of image
//nSrcStep_Dev is the step(or pitch) value of the Source image
//subtracts min value from all SOURCE pixels
//call this function with 512*512=262144 threads
__global__ void kernel2( Npp8u *pSrc_Dev, int nWidth, int nHeight, int nSrcStep_Dev, Npp8u *pMin_Dev)
{
	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	int index = (tid / nWidth)*nSrcStep_Dev + tid % nWidth;

	pSrc_Dev[index] -= *pMin_Dev;
}

//pSrc_Dev is pointer to image
//nWidth & nHeight are dimensions of image
//nSrcStep_Dev is the step(or pitch) value of the Source image
//nConstant & scaleFact are used for multiplying each pixel 
//multiplies all SOURCE pixels with nConstant/scaleFact
//call this function with 512*512=262144 threads
__global__ void kernel3(Npp8u * pSrc_Dev, int nWidth, int nHeight, int nSrcStep_Dev, Npp8u *pMax, Npp8u* pMin)
{
	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	int index = (tid / nWidth)*nSrcStep_Dev + tid % nHeight;

	pSrc_Dev[index]*= 255.0f / (*pMax-*pMin);

	/*pSrc_Dev[index] *= nConstant;
	pSrc_Dev[index] /= scaleFact;*/
}


// ###################### kernel callers ###################### //
void CallKernel1(int blockCount, int blockSize, Npp8u *pSrc_Dev, int nWidth, int nHeight, int nSrcStep_Dev)
{
	kernel1 <<< blockCount, blockSize >>> (pSrc_Dev, nWidth, nHeight, nSrcStep_Dev);
	ErrorCheck("kernel1");
}

void CallKernel2(int blockCount, int blockSize, Npp8u *pSrc_Dev, int nWidth, int nHeight, int nSrcStep_Dev, Npp8u *pMin_Dev)
{
	kernel2 << <blockCount, blockSize >> > (pSrc_Dev, nWidth, nHeight, nSrcStep_Dev, pMin_Dev);
	ErrorCheck("kernel2");
}

void CallKernel3(int blockCount, int blockSize, Npp8u * pSrc_Dev, int nWidth, int nHeight, int nSrcStep_Dev, Npp8u *pMax_Dev, Npp8u *pMin_Dev)
{
	kernel3 << < blockCount, blockSize >> > (pSrc_Dev, nWidth, nHeight, nSrcStep_Dev, pMax_Dev, pMin_Dev);
	ErrorCheck("kernel3");
}

