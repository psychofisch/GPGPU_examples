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

inline __device__ bool operator==(float3& lhs, float3& rhs)
{
	if (lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z)
		return true;
	else
		return false;
}

__device__ float4 calculatePressure(float4* position, uint index, uint numberOfParticles, float interactionRadius);

__global__ void particleUpdate(
	float4* position, 
	float4* positionOut,
	float4* velocity, 
	const float dt, 
	const float interactionRadius, 
	const float3 gravity,
	const float3 dimension,
	const uint numberOfParticles)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= numberOfParticles)
		return;

	float3 particlePosition = make_float3(position[index]);
	float3 particleVelocity = make_float3(velocity[index]);
	float4 particlePressure = calculatePressure(position, index, numberOfParticles, interactionRadius);

	particleVelocity += (gravity + make_float3(particlePressure)) * dt;

	// static collision
	//TODO: write some kind of for-loop
	if ((particlePosition.x + particleVelocity.x * dt > dimension.x && particleVelocity.x > 0.f) || (particlePosition.x + particleVelocity.x * dt < 0.f && particleVelocity.x < 0.f))
	{
		if (particlePosition.x + particleVelocity.x * dt < 0.f)
			particlePosition.x = 0.f;
		else
			particlePosition.x = dimension.x;

		particleVelocity.x *= -.3f;
	}

	if ((particlePosition.y + particleVelocity.y * dt  > dimension.y && particleVelocity.y > 0.f) || (particlePosition.y + particleVelocity.y * dt < 0.f && particleVelocity.y < 0.f))
	{
		if (particlePosition.y + particleVelocity.y * dt < 0.f)
			particlePosition.y = 0.f;
		else
			particlePosition.y = dimension.y;

		particleVelocity.y *= -.3f;
	}

	if ((particlePosition.z + particleVelocity.z * dt > dimension.z && particleVelocity.z > 0.f) || (particlePosition.z + particleVelocity.z * dt < 0.f && particleVelocity.z < 0.f))
	{
		if (particlePosition.z + particleVelocity.z * dt < 0.f)
			particlePosition.z = 0.f;
		else
			particlePosition.z = dimension.z;

		particleVelocity.z *= -.3f;
	}
	// *** sc

	// particleVelocity += dt * particleVelocity * -0.01f;//damping
	particlePosition += particleVelocity * dt;

	positionOut[index] = make_float4(particlePosition);
	velocity[index] = make_float4(particleVelocity);
}

__device__ float4 calculatePressure(float4* position, uint index, uint numberOfParticles, float interactionRadius)
{
	float3 particlePosition = make_float3(position[index]);

	float4 pressureVec = make_float4(0.f);
	for (uint i = 0; i < numberOfParticles; i++)
	{
		if (index == i)
			continue;

		float3 dirVec = particlePosition - make_float3(position[i]);
		float dist = length(dirVec);//TODO: maybe use half_length

		if (dist > interactionRadius * 1.0f)
			continue;

		float pressure = 1.f - (dist / interactionRadius);
		////float pressure = amplitude * exp(-dist / interactionRadius);

		pressureVec += make_float4(pressure * normalize(dirVec));
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
	const float interactionRadius,
	const float3 gravity,
	const float3 dimension,
	const uint numberOfParticles)
{
	cudaDeviceProp devProp;
	int device;
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&devProp, device);

	int num = 1;
	int threads = numberOfParticles;
	int maxThreads = devProp.maxThreadsPerBlock;

	if (numberOfParticles > (uint)maxThreads)
	{
		num = (int)ceilf(float(numberOfParticles) / maxThreads);
		threads = maxThreads;
	}

	particleUpdate<<< num, threads >>>(position, positionOut, velocity, dt, interactionRadius, gravity, dimension, numberOfParticles);
}
