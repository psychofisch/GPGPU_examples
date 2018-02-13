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

float4 calculatePressure(float4* position, uint index, uint numberOfParticles, float smoothingWidth);

__global__ void particleUpdate(
	float4* position, 
	float4* velocity, 
	const float dt, 
	const float smoothingWidth, 
	const float4 gravity,
	const float4 dimension,
	const uint numberOfParticles)
{
	const uint index = threadIdx.x;

	if (index >= numberOfParticles)
		return;

	float4 particlePosition = position[index];
	float4 particleVelocity = velocity[index];
	float4 particlePressure = calculatePressure(position, index, numberOfParticles, smoothingWidth);

	if (particlePosition.x <= dimension.x || particlePosition.x >= 0.f
		|| particlePosition.y <= dimension.y || particlePosition.y >= 0.f
		|| particlePosition.z <= dimension.z || particlePosition.z >= 0.f)
		particleVelocity += (gravity + particlePressure) * dt;

	// static collision
	//TODO: write some kind of for-loop
	if ((particlePosition.x > dimension.x && particleVelocity.x > 0.f) || (particlePosition.x < 0.f && particleVelocity.x < 0.f))
	{
		particleVelocity.x *= -.3f;
	}

	if ((particlePosition.y > dimension.y && particleVelocity.y > 0.f) || (particlePosition.y < 0.f && particleVelocity.y < 0.f))
	{
		particleVelocity.y *= -.3f;
	}

	if ((particlePosition.z > dimension.z && particleVelocity.z > 0.f) || (particlePosition.z < 0.f && particleVelocity.z < 0.f))
	{
		particleVelocity.z *= -.3f;
	}
	// *** sc

	// particleVelocity += dt * particleVelocity * -0.01f;//damping
	particlePosition += particleVelocity * dt;

	position[index] = particlePosition;
	velocity[index] = particleVelocity;
}

//__device__ float4 calculatePressure(float4* position, uint index, uint numberOfParticles, float smoothingWidth)
__device__ float4 calculatePressure(float4* position, uint index, uint numberOfParticles, float smoothingWidth)
{
	float4 particlePosition = position[index];

	float4 pressureVec;
	for (uint i = 0; i < numberOfParticles; i++)
	{
		if (index == i)
			continue;

		float4 dirVec = particlePosition - position[i];
		float dist = length(dirVec);//TODO: maybe use half_length

		if (dist > smoothingWidth * 1.0f)
			continue;

		float pressure = 1.f - (dist / smoothingWidth);
		////float pressure = amplitude * exp(-dist / smoothingWidth);

		pressureVec += pressure * normalize(dirVec);
		//// pressureVec += vec4(dirVec, 0.f);

		pressureVec.w = dist;

		//break;
	}

	return pressureVec;
}

extern "C" void cudaUpdate(
	float4* position,
	float4* velocity,
	const float dt,
	const float smoothingWidth,
	const float4 gravity,
	const float4 dimension,
	const unsigned int numberOfParticles)
{
	cudaDeviceProp devProp;
	int device;
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&devProp, device);

	int num = 1;
	int threads = numberOfParticles;

	if (numberOfParticles > devProp.maxThreadsPerBlock)
	{
		num = ceilf(float(numberOfParticles) / devProp.maxThreadsPerBlock);
		threads = devProp.maxThreadsPerBlock;
	}

	particleUpdate<<< num, threads >>>(position, velocity, dt, smoothingWidth, gravity, dimension, numberOfParticles);
}
