#include <npp.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include "cpuFunctions.h"
#include <thrust\device_ptr.h>
// CPU FUNCTIONS

//img is pointer to image, width & height are dimensions of image
// minMax is host pointer to min & max pointer pair of the image
//changes min&max pointers' values, returns void
void CPUfunc1(Npp8u *img, int width, int height, thrust::pair<Npp8u*,Npp8u*>* minMax)
{
	//use thrust min max to find min & max on HOST
	minMax->swap(thrust::minmax_element(img, img + width * height));
}

//img is pointer to image, width & height are dimensions of image
//min is minimum value of image
//changes pixels' values, returns void
void CPUfunc2(Npp8u *img, Npp8u *dstImg, int width, int height, Npp8u min)
{
	//go through all the pixels in image, subtract min from each of them--with thrust::transfrom(,,,thrust::minus<npp8u>())
	thrust::transform(img, img + width * height, 
		thrust::make_constant_iterator(min), dstImg, thrust::minus<Npp8u>());//subtracts m from all img, stores the result in img
}

//img is pointer to image, width & height are dimensions of image
//changes pixels values, returns void
void CPUfunc3(Npp8u *img, int width, int height, int maxSubMin)
{
	//go through all the pixels in image, multiply all the pixels with the nConstant
	//and then divide them by nScaleFactor -1 to achieve: 255/(nMax_Host-nMinHost)))--with thrust::transfrom(,,,thrust::multiplies<npp8u>())

	float c = 255.0f / maxSubMin;
	thrust::transform(img, img + width * height,
		thrust::make_constant_iterator(c), img, thrust::multiplies<float>());//multiplies all pixels of img, stores the result in img
}
