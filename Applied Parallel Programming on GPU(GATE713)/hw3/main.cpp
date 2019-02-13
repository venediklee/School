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
#include <thrust\device_vector.h>

//--------------------- cpu timer
double PCFreq = 0.0;
__int64 CounterStart = 0;

void StartCounter()
{
	LARGE_INTEGER li;
	if (!QueryPerformanceFrequency(&li))
		std::cout << "QueryPerformanceFrequency failed!\n";

	PCFreq = double(li.QuadPart) / 1000.0;

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
	thrust::pair<Npp8u*, Npp8u*> minMax_host;

	// Load image to the host.
	std::cout << "Load PGM file." << std::endl;
	pSrc_Host = LoadPGM("C:\\Users\\Bright Lord\\Desktop\\713\\hw3\\lena_before.pgm", nWidth, nHeight, nMaxGray);
	pDst_Host = new Npp8u[nWidth * nHeight];

	

	


	//	##################### cpu1 ##################### //
	StartCounter();
	CPUfunc1(pSrc_Host, nWidth, nHeight, &minMax_host);
	//std::cout << "min & max host values are: ->" <<(int) *minMax_host.first << "  &&  "<< (int)*minMax_host.second<<std::endl;
	//std::cout << "Total CPU Time Spent: " << GetCounter() << std::endl;
	//	##################### cpu2 ##################### //
	CPUfunc2(pSrc_Host, pDst_Host, nWidth, nHeight, *minMax_host.first);
	//std::cout << "Total CPU Time Spent: " << GetCounter() << std::endl;
	//	##################### cpu3 ##################### //
	CPUfunc3(pDst_Host, nWidth, nHeight, (*minMax_host.second) - (*minMax_host.first));
	std::cout << "Total CPU Time Spent: " << GetCounter() << std::endl;
	//---------------------------------------------------------------
	//###################### Output the result cpu image. ###################### 
	std::cout << "Output the PGM file." << std::endl;
	WritePGM("C:\\Users\\Bright Lord\\Desktop\\713\\hw3\\Thrust_CPU_lena_after.pgm", pDst_Host, nWidth, nHeight, nMaxGray);



	// ------------------GPU ---------------------- \\

	// Device parameter declarations.
	//thrust::device_vector<Npp8u*> pSrc_Dev(nWidth*nHeight);
	Npp8u    nMin_Host, nMax_Host;
	NppiSize oROI;
	int		 nSrcStep_Dev, nDstStep_Dev;
	Npp8u temp = 0;
	//thrust::pair<Npp8u*, Npp8u*> minMax_dev;//=thrust::make_pair(&temp,&temp);
	
	//benchmarking
	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	
	std::cout << "Copy image from host to device." << std::endl;
	// Copy the image from the host to GPU
	/*oROI.width = nWidth;
	oROI.height = nHeight;
	pSrc_Dev = nppiMalloc_8u_C1(nWidth, nHeight, &nSrcStep_Dev);

	cudaMemcpy2D(pSrc_Dev, nSrcStep_Dev, pSrc_Host, nWidth, nWidth, nHeight, cudaMemcpyHostToDevice);*/
	//pSrc_Dev= thrust::device_malloc<Npp8u>(nWidth*nHeight);
	//thrust::copy(pSrc_Host, pSrc_Host + nWidth * nHeight, pSrc_Dev.begin());

	std::cout << "Process the image on GPU." << std::endl;
	
	//	##################### kernel1 ##################### //
	//callKernel1(pSrc_Dev,nWidth, nHeight, &minMax_dev);
	//minMax_dev.swap(thrust::minmax_element(pSrc_Dev, pSrc_Dev+ nWidth * nHeight));
	//CPUfunc1(pSrc_Dev, nWidth, nHeight, &minMax_dev);
	//minMax_dev = thrust::make_pair<Npp8u*,Npp8u*>(thrust::min_element(pSrc_Dev, pSrc_Dev + nWidth * nHeight),
		//thrust::max_element(pSrc_Dev, pSrc_Dev + nWidth * nHeight));
	//minMax_dev.swap(thrust::minmax_element(pSrc_Dev, pSrc_Dev+ nWidth * nHeight));
	//CPUfunc1(thrust::device_pointer_cast<Npp8u*>( pSrc_Dev), nWidth, nHeight, &minMax_host);
	//thrust::swap(minMax_dev,thrust::minmax_element(pSrc_Dev.begin(), pSrc_Dev.end()));
	std::cout << "kernel1 done" << std::endl;

	//	##################### kernel2 ##################### //
	//kernel2(pSrc_Dev,nWidth,nHeight, *minMax_dev.first);
	std::cout << "kernel2 done" << std::endl;

	//	##################### kernel3 ##################### //
	//kernel3(pSrc_Dev,nWidth,nHeight, (*minMax_host.second) - (*minMax_host.first));
	std::cout << "kernel3 done" << std::endl;

	//-------------------
	// Copy result back to the host.
	std::cout << "Work done! Copy the result back to host." << std::endl;
	//thrust::copy(pSrc_Dev, pSrc_Dev+nWidth*nHeight, pDst_Host);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float cudaTotalTime;
	cudaEventElapsedTime(&cudaTotalTime, start, stop);
	cudaEventDestroy(start); cudaEventDestroy(stop);
	std::cout << "Total GPU Time Spent: " << cudaTotalTime << std::endl;
	 //###################### GPU result stored on pDst_host ###################### 
	WritePGM("C:\\Users\\Bright Lord\\Desktop\\713\\hw3\\Thrust_GPU_lena_after.pgm", pDst_Host, nWidth, nHeight, nMaxGray);

	// Clean up.
	std::cout << "Clean up." << std::endl;
	delete[] pSrc_Host;
	delete[] pDst_Host;

	//thrust::device_free(pSrc_Dev);
	
	

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

