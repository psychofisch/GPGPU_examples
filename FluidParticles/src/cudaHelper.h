#pragma once

#include <cuda_runtime.h>
#include <helper_cuda.h>

class cudaHelper
{
public:
	cudaHelper();
	~cudaHelper();

	cudaError setupCUDA(int argc, const char ** argv);
};

