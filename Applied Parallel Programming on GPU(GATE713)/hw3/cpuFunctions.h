//#pragma once

#include <thrust\extrema.h>//minmax
#include <thrust\iterator\constant_iterator.h>//constant iterator etc.
#include <thrust\transform.h>
#include <thrust\host_vector.h>

void CPUfunc1(Npp8u * img, int width, int height, thrust::pair<Npp8u*, Npp8u*>* minMax);

void CPUfunc2(Npp8u * img, Npp8u *dstImg, int width, int height, Npp8u min);

void CPUfunc3(Npp8u * img, int width, int height, int maxSubMin);
