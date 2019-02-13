#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include "npp.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust\device_vector.h>
#include <thrust\copy.h>
#include <thrust\extrema.h>
#include <thrust\iterator\constant_iterator.h>
#include <thrust\swap.h>

void ErrorCheck(char * c);



 
 void kernel1(Npp8u * img, int width, int height, thrust::pair<Npp8u*, Npp8u*>* minMax_dev);

 void kernel2(Npp8u * img, int width, int height, Npp8u pMin_Dev);

 void kernel3(Npp8u * img, int width, int height, Npp8u maxSubMinDev);


 void callKernel1(Npp8u * img, int width, int height, thrust::pair<Npp8u*, Npp8u*>* minMax_dev);


