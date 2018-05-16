#pragma once

// CUDA runtime
#include <cuda_runtime.h>

struct MinMaxDataCuda
{
	float4 min, max;

	/*MinMaxData operator+(float4 p_)
	{
	MinMaxData tmp;
	tmp.min = this->min + p_;
	tmp.max = this->max + p_;
	return tmp;
	}

	MinMaxData operator=(const MinMaxData& other)
	{
	this->min = other.min;
	this->max = other.max;
	return *this;
	}*/
};
