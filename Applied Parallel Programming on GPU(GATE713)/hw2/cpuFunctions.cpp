#include <npp.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include "cpuFunctions.h"
// CPU FUNCTIONS

//img is pointer to image, width & height are dimensions of image
// min & max are host pointers to min & max values of the image
//changes min&max pointers' values, returns void
void CPUfunc1(Npp8u *img, int width, int height, Npp8u *min, Npp8u *max)
{
	//set min&max to default value
	*min = img[0];
	*max = img[0];
	//std::cout << "min is: " << (int) img[0] << std::endl << "max is: " <<(int) img[0] << std::endl;
	//go through all the pixels in image, set min & max 
	for (int i = 0; i < height*width; i++)
	{
		if (img[i] > *max) *max = img[i];
		if (img[i] < *min) *min = img[i];
	}
	//std::cout << "@End of func1 min is: " << (int) *min << std::endl << "@End of func1 max is: " <<(int) *max << std::endl;
}

//img is pointer to image, width & height are dimensions of image
//min is minimum value of image
//changes pixels' values, returns void
void CPUfunc2(Npp8u *img, int width, int height, Npp8u min)
{
	//std::cout << "@Func2 min is: " << (int)min << std::endl;
	//go through all the pixels in image, subtract min from each of them
	for (int i = 0; i < height*width; i++) img[i] -= min; //& 0xF3;		
}

//img is pointer to image, width & height are dimensions of image
// nConstant & nScaleFactor are used for multiplying etc. (nScaleFactor within this function is already decremented)
//changes pixels values, returns void
void CPUfunc3(Npp8u *img, int width, int height, int maxSubMin)
{
	//go through all the pixels in image, multiply all the pixels with the nConstant 
	//and then divide them by nScaleFactor -1 to achieve: 255/(nMax_Host-nMinHost)))
	for (int i = 0; i < height*width; i++)
	{
		img[i] *= 255.0f / maxSubMin ;
		//img[i] *= nConstant / nScaleFactor;
	}
}
