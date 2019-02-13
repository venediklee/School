#include "kernels.cuh"


//KERNELS


//used for error checking
void ErrorCheck(char *c)
{
	if (cudaGetLastError() != 0)
	{
		std::cout << ("error @ %s -> %s \n", c, cudaGetErrorString(cudaGetLastError())) << std::endl;
		//exit(-1);
	}
}



void kernel1(Npp8u *img, int width, int height,  thrust::pair<Npp8u*, Npp8u*> *minMax_dev)
{
	minMax_dev->swap(thrust::minmax_element(img, img + width * height));

}



void kernel2(Npp8u *img,int width,int height, Npp8u pMin_Dev)
{
	//go through all the pixels in image, subtract min from each of them--with thrust::transfrom(,,,thrust::minus<npp8u>())
	thrust::transform(img, img+width*height,
		thrust::make_constant_iterator(pMin_Dev), img, thrust::minus<Npp8u>());//subtracts m from all img, stores the result in img
}


void kernel3(Npp8u *img,int width, int height, Npp8u maxSubMinDev)
{
	//go through all the pixels in image, multiply all the pixels with the nConstant
	//and then divide them by nScaleFactor -1 to achieve: 255/(nMax_Host-nMinHost)))--with thrust::transfrom(,,,thrust::multiplies<npp8u>())

	float c = 255.0f / maxSubMinDev;
	thrust::transform(img, img+width*height,
		thrust::make_constant_iterator(c),img, thrust::multiplies<float>());//multiplies all pixels of img, stores the result in img
}


void callKernel1(Npp8u * img, int width, int height, thrust::pair<Npp8u*, Npp8u*>* minMax_dev)
{
	//kernel1(img, width, height, minMax_dev);
}
