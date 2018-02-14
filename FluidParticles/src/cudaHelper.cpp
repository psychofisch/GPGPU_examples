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

void cudaHelper::cudaUpdate(float4 * positions, float4 * velocity, const float dt, const float smoothingWidth, const ofVec4f gravity, const ofVec4f dimension, const unsigned int numberOfParticles)
{
	dim3 grid(1, 0, 0);
	dim3 threads(numberOfParticles, 0, 0);

	float4 cudaGravity = make_float4(gravity.x, gravity.y, gravity.z, 0);
	float4 cudaDimension = make_float4(dimension.x, dimension.y, dimension.z, 0);

	particleUpdate<<< grid, threads >>> (positions, velocity, dt, smoothingWidth, cudaGravity, cudaDimension, numberOfParticles);
}
