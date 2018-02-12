// System includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>
 
typedef unsigned int uint;

float3 calculatePressure(float3* positions, uint1 index, uint numberOfParticles, float smoothingWidth);

__global__ void particleUpdate(
	float3* positions, 
	float3* velocity, 
	const float dt, 
	const float smoothingWidth, 
	const float3 gravity,
	const float4 dimension,
	const uint numberOfParticles)
{

}

float3 calculatePressure(float3* positions, uint index, uint numberOfParticles, float smoothingWidth)
{
	float3 particlePosition = positions[index];

	float3 pressureVec = make_float3(0,0,0);
	for (uint i = 0; i < numberOfParticles; i++)
	{
		if (index == i)
			continue;

		//float3 dirVec = particlePosition - positions[i];
		//float dist = length(dirVec);//TODO: maybe use half_length

		//if (dist > smoothingWidth * 1.0f)
		//	continue;

		//float pressure = 1.f - (dist / smoothingWidth);
		////float pressure = amplitude * exp(-dist / smoothingWidth);

		//pressureVec += (float4)(pressure * normalize(dirVec), 0.f);
		//// pressureVec += vec4(dirVec, 0.f);

		//pressureVec.w = dist;

		//break;
	}

	return pressureVec;
}
