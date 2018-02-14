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
	float4* positionOut,
	float4* velocity, 
	const float dt, 
	const float smoothingWidth, 
	const float4 gravity,
	const float4 dimension,
	const uint numberOfParticles)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= numberOfParticles)
		return;

	float4 particlePosition = position[index];
	float4 particleVelocity = velocity[index];
	//float4 particlePressure = calculatePressure(position, index, numberOfParticles, smoothingWidth);
	float4 particlePressure = make_float4(0.f);
	if (isnan(particlePosition.x))
		printf("%u Pos = NAN\n", index);

	if (isnan(particleVelocity.x))
		printf("%u Vel = NAN\n", index);

	if (isnan(particlePressure.x))
		printf("%u Pressure = NAN\n", index);

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

	positionOut[index] = particlePosition;
	velocity[index] = particleVelocity;
}

__device__ float4 calculatePressure(float4* position, uint index, uint numberOfParticles, float smoothingWidth)
{
	float4 particlePosition = position[index];

	float4 pressureVec = make_float4(0.f);
	for (uint i = 0; i < numberOfParticles; i++)
	{
		if (index == i)
			continue;

		float4 dirVec4 = particlePosition - position[i];
		float3 dirVec = make_float3(dirVec4.x, dirVec4.x, dirVec4.z);
		float dist = length(dirVec);//TODO: maybe use half_length

		if (dist > smoothingWidth * 1.0f)
			continue;

		float pressure = 1.f - (dist / smoothingWidth);
		////float pressure = amplitude * exp(-dist / smoothingWidth);

		pressureVec += make_float4(pressure * normalize(dirVec), 0);
		//// pressureVec += vec4(dirVec, 0.f);

		pressureVec.w += pressure;

		//break;
	}

	return pressureVec;
}

extern "C" void cudaUpdate(
	float4* position,
	float4* positionOut,
	float4* velocity,
	const float dt,
	const float smoothingWidth,
	const float4 gravity,
	const float4 dimension,
	const uint numberOfParticles)
{
	cudaDeviceProp devProp;
	int device;
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&devProp, device);

	int num = 1;
	int threads = numberOfParticles;
	int maxThreads = devProp.maxThreadsPerBlock;

	if (numberOfParticles > maxThreads)
	{
		num = ceilf(float(numberOfParticles) / maxThreads);
		threads = maxThreads;
	}

	particleUpdate<<< num, threads >>>(position, positionOut, velocity, dt, smoothingWidth, gravity, dimension, numberOfParticles);
}
