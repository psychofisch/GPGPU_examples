#pragma once

#include <ofVec4f.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>

extern "C" void particleUpdate(
	float4* positions,
	float4* velocity,
	const float dt,
	const float smoothingWidth,
	const float4 gravity,
	const float4 dimension,
	const unsigned int numberOfParticles);

class cudaHelper
{
public:
	cudaHelper();
	~cudaHelper();

	cudaError setupCUDA(int argc, const char ** argv);
	void cudaUpdate(
		float4* positions,
		float4* velocity,
		const float dt,
		const float smoothingWidth,
		const ofVec4f gravity,
		const ofVec4f dimension,
		const unsigned int numberOfParticles);
};

