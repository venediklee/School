#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include "npp.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

void ErrorCheck(char * c);



void CallKernel1(int blockCount, int blockSize, Npp8u *pSrc_Dev, int nWidth, int nHeight, int nSrcStep_Dev);
void CallKernel2(int blockCount, int blockSize, Npp8u *pSrc_Dev, int nWidth, int nHeight, int nSrcStep_Dev, Npp8u *pMin_Dev);
void CallKernel3(int blockCount, int blockSize, Npp8u *pSrc_Dev, int nWidth, int nHeight, int nSrcStep_Dev, Npp8u *pMax_Dev, Npp8u * pMin_Dev);
