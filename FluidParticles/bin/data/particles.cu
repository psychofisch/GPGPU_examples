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
#include <helper_math.h> 

typedef unsigned int uint;

float4 calculatePressure(float4* positions, uint index, uint numberOfParticles, float smoothingWidth);

__global__ void particleUpdate(
	float4* positions, 
	float4* velocity, 
	const float dt, 
	const float smoothingWidth, 
	const float4 gravity,
	const float4 dimension,
	const uint numberOfParticles)
{
	const uint tid = threadIdx.x;

	positions[tid] += gravity * dt;
}

__global__ float4 calculatePressure(float4* positions, uint index, uint numberOfParticles, float smoothingWidth)
{
	float4 particlePosition = positions[index];

	float4 pressureVec;
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

extern "C" void cudaUpdate(
	float4* positions,
	float4* velocity,
	const float dt,
	const float smoothingWidth,
	const float4 gravity,
	const float4 dimension,
	const unsigned int numberOfParticles)
{
	dim3 grid(1, 0, 0);
	dim3 threads(numberOfParticles, 0, 0);
	particleUpdate<<< grid, threads >>>(positions, velocity, dt, smoothingWidth, gravity, dimension, numberOfParticles);
}
