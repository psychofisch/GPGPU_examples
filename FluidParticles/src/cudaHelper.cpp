#include "cudaHelper.h"



cudaHelper::cudaHelper()
{
}


cudaHelper::~cudaHelper()
{
}

cudaError cudaHelper::setupCUDA(int argc, const char ** argv)
{
	// use command-line specified CUDA device, otherwise use device with highest Gflops/s
	findCudaDevice(argc, argv);


	return cudaSuccess;
}
