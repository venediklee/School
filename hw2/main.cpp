/*
* Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

// This example implements the contrast adjustment on an 8u one-channel image by using
// Nvidia Performance Primitives (NPP). 
// Assume pSrc(i,j) is the pixel value of the input image, nMin and nMax are the minimal and 
// maximal values of the input image. The adjusted image pDst(i,j) is computed via the formula:
// pDst(i,j) = (pSrc(i,j) - nMin) / (nMax - nMin) * 255 
//
// The code flow includes five steps:
// 1) Load the input image into the host array;
// 2) Allocate the memory space on the GPU and copy data from the host to GPU;
// 3) Call NPP functions to adjust the contrast;
// 4) Read data back from GPU to the host;
// 5) Output the result image and clean up the memory.

#include <iostream>
#include <fstream>
#include <sstream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "kernels.cuh"
#include "cpuFunctions.h"

#include <windows.h>



//--------------------- cpu timer
double PCFreq = 0.0;
__int64 CounterStart = 0;

void StartCounter()
{
	LARGE_INTEGER li;
	if (!QueryPerformanceFrequency(&li))
		std::cout << "QueryPerformanceFrequency failed!\n";

	PCFreq = double(li.QuadPart)/1000.0;

	QueryPerformanceCounter(&li);
	CounterStart = li.QuadPart;
}
double GetCounter()
{
	LARGE_INTEGER li;
	QueryPerformanceCounter(&li);
	return double(li.QuadPart - CounterStart) / PCFreq;
}
//-------------------


// Function declarations.
Npp8u * LoadPGM(char * sFileName, int & nWidth, int & nHeight, int & nMaxGray);
void WritePGM(char * sFileName, Npp8u * pDst_Host, int nWidth, int nHeight, int nMaxGray);

// Main function.
int main(int argc, char ** argv)
{
	// Host parameter declarations.	
	Npp8u * pSrc_Host, *pDst_Host;
	int   nWidth, nHeight, nMaxGray;

	// Load image to the host.
	std::cout << "Load PGM file." << std::endl;
	pSrc_Host = LoadPGM("C:\\Users\\Bright Lord\\Desktop\\713\\hw2\\Assgn2_Files\\Code\\lena_before.pgm", nWidth, nHeight, nMaxGray);
	pDst_Host = new Npp8u[nWidth * nHeight];

	// Device parameter declarations.
	Npp8u	 * pSrc_Dev, *pDst_Dev;
	Npp8u    * pMin_Dev, *pMax_Dev;
	Npp8u    nMin_Host, nMax_Host;
	NppiSize oROI;
	int		 nSrcStep_Dev, nDstStep_Dev;

	//benchmarking
	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);


	//	##################### cpu1 ##################### //
	StartCounter();
	CPUfunc1(pSrc_Host, nWidth, nHeight, (Npp8u *)&nMin_Host, (Npp8u *)&nMax_Host);
	//	##################### cpu2 ##################### //
	CPUfunc2(pSrc_Host, nWidth, nHeight, nMin_Host);
	//	##################### cpu3 ##################### //
	StartCounter();
	CPUfunc3(pSrc_Host, nWidth, nHeight, nMax_Host - nMin_Host);
	std::cout << "Total CPU Time Spent: " << GetCounter() << std::endl;
	//---------------------------------------------------------------



	// Copy the image from the host to GPU
	oROI.width = nWidth;
	oROI.height = nHeight;
	pSrc_Dev = nppiMalloc_8u_C1(nWidth, nHeight, &nSrcStep_Dev);
	pDst_Dev = nppiMalloc_8u_C1(nWidth, nHeight, &nDstStep_Dev);
	std::cout << "Copy image from host to device." << std::endl;

	cudaEventRecord(start, 0);
	
	cudaMemcpy2D(pSrc_Dev, nSrcStep_Dev, pSrc_Host, nWidth, nWidth, nHeight, cudaMemcpyHostToDevice);
	cudaMemcpy2D(pDst_Dev, nDstStep_Dev, pSrc_Host, nWidth, nWidth, nHeight, cudaMemcpyHostToDevice);

	std::cout << "Process the image on GPU." << std::endl;

	//CALL KERNEL1 with 262144/4=65536 threads // each thread checks 4 elements etc
	int blockSize = 1024;//!!max limit!!
	int blockCount = 64;

	//	##################### kernel1 ##################### //
	CallKernel1(blockCount, blockSize, pSrc_Dev, nWidth, nHeight, nSrcStep_Dev);//min @psrc[0] max @psrc[131072](nonpadded)
	//	##################### kernel1 ##################### //

	//allocate then copy min&max values
	cudaMalloc((void**)(&pMin_Dev), sizeof(Npp8u)); 
	cudaMalloc((void **)(&pMax_Dev), sizeof(Npp8u));

	cudaMemcpy(pMin_Dev, pSrc_Dev, sizeof(Npp8u), cudaMemcpyDeviceToDevice);
	cudaMemcpy(pMax_Dev, pSrc_Dev + ((131072) / nWidth)*nSrcStep_Dev + (131072) % nWidth, sizeof(Npp8u), cudaMemcpyDeviceToDevice);

	// Call SubC primitive.
	// Replace this line with your KERNEL2 call (KERNEL2: your kernel subtracting the nMin_Host from all the pixels)

	//call kernel2 with 512*512=262144 threads // each threads subtracts min value from a particular pixel
	blockSize = 1024;//!!max value!!
	blockCount = 256;
	//re-set the pSrc_Dev since we changed values @kernel1 -- doesnt work, copy pDst_Dev instead
	//cudaMemcpy2D(pSrc_Dev, nSrcStep_Dev, pSrc_Host, nWidth, nWidth, nHeight, cudaMemcpyHostToDevice);

	//	##################### kernel2 ##################### //
	CallKernel2(blockCount, blockSize, pDst_Dev, nWidth, nHeight, nDstStep_Dev, pMin_Dev);
	//	##################### kernel2 ##################### //

	// Call MulC primitive.
	//TODO Replace this line with your KERNEL3 call (KERNEL3: your kernel multiplying all the pixels with the nConstant and then dividing them by nScaleFactor -1 to achieve: 255/(nMax_Host-nMinHost)))
	
	//call kernel3 with 512*512=262144 threads // each threads multiplies a particular pixel with 255.0f/(nMax_Host-nMinHost)
	blockSize = 1024;//!!max size!!
	blockCount = 256;

	//	##################### kernel3 ##################### //
	CallKernel3(blockCount, blockSize, pDst_Dev, nWidth, nHeight, nDstStep_Dev, pMax_Dev, pMin_Dev);
	//	##################### kernel3 ##################### //

	//-------------------
	// Copy result back to the host.
	std::cout << "Work done! Copy the result back to host." << std::endl;
	cudaMemcpy2D(pDst_Host, nWidth * sizeof(Npp8u), pDst_Dev, nDstStep_Dev, nWidth * sizeof(Npp8u), nHeight, cudaMemcpyDeviceToHost);
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float cudaTotalTime;
	cudaEventElapsedTime(&cudaTotalTime, start, stop);
	cudaEventDestroy(start); cudaEventDestroy(stop);
	std::cout << "Total GPU Time Spent: " << cudaTotalTime << std::endl;
	//###################### Output the result image.(CPU result stored on pSrc_Host // GPU result stored on pDst_host ###################### 
	std::cout << "Output the PGM file." << std::endl;
	WritePGM("C:\\Users\\Bright Lord\\Desktop\\713\\hw2\\Assgn2_Files\\Code\\CPU_lena_after.pgm", pSrc_Host, nWidth, nHeight, nMaxGray);
	WritePGM("C:\\Users\\Bright Lord\\Desktop\\713\\hw2\\Assgn2_Files\\Code\\GPU_lena_after.pgm", pDst_Host, nWidth, nHeight, nMaxGray);

	// Clean up.
	std::cout << "Clean up." << std::endl;
	delete[] pSrc_Host;
	delete[] pDst_Host;

	nppiFree(pSrc_Dev);
	nppiFree(pDst_Dev);
	nppiFree(pMin_Dev);
	nppiFree(pMax_Dev);

	
	

	//cudaFree(&cudaTotalTime);

	return 0;
}

// Disable reporting warnings on functions that were marked with deprecated.
#pragma warning( disable : 4996 )

// Load PGM file.
Npp8u *
LoadPGM(char * sFileName, int & nWidth, int & nHeight, int & nMaxGray)
{
	char aLine[256];
	FILE * fInput = fopen(sFileName, "r");
	if (fInput == 0)
	{
		perror("Cannot open file to read");
		exit(EXIT_FAILURE);
	}
	// First line: version
	fgets(aLine, 256, fInput);
	std::cout << "\tVersion: " << aLine;
	// Second line: comment
	fgets(aLine, 256, fInput);
	std::cout << "\tComment: " << aLine;
	fseek(fInput, -1, SEEK_CUR);
	// Third line: size
	fscanf(fInput, "%d", &nWidth);
	std::cout << "\tWidth: " << nWidth;
	fscanf(fInput, "%d", &nHeight);
	std::cout << " Height: " << nHeight << std::endl;
	// Fourth line: max value
	fscanf(fInput, "%d", &nMaxGray);
	std::cout << "\tMax value: " << nMaxGray << std::endl;
	while (getc(fInput) != '\n');
	// Following lines: data
	Npp8u * pSrc_Host = new Npp8u[nWidth * nHeight];
	for (int i = 0; i < nHeight; ++i)
		for (int j = 0; j < nWidth; ++j)
			pSrc_Host[i*nWidth + j] = fgetc(fInput);
	fclose(fInput);

	return pSrc_Host;
}

// Write PGM image.
void
WritePGM(char * sFileName, Npp8u * pDst_Host, int nWidth, int nHeight, int nMaxGray)
{
	FILE * fOutput = fopen(sFileName, "w+");
	if (fOutput == 0)
	{
		perror("Cannot open file to read");
		exit(EXIT_FAILURE);
	}
	char * aComment = "# Created by NPP";
	fprintf(fOutput, "P5\n%s\n%d %d\n%d\n", aComment, nWidth, nHeight, nMaxGray);
	for (int i = 0; i < nHeight; ++i)
		for (int j = 0; j < nWidth; ++j)
			fputc(pDst_Host[i*nWidth + j], fOutput);
	fclose(fOutput);
}

