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

#include "ParticleDefinitions.h"

//inline __device__ bool operator==(float3& lhs, float3& rhs)
//{
//	if (lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z)
//		return true;
//	else
//		return false;
//}

__device__ float4 calculatePressure(const float4* __restrict__ positions, const float4* __restrict__ velocity, uint index, float3 pos, float3 vel, uint numberOfParticles, SimulationData simData);

__global__ void particleUpdate(
	const float4* __restrict__ positions,
	float4* __restrict__ positionOut,
	float4* __restrict__ velocity,
	const float dt, 
	const float3 gravity,
	const float3 position,
	const float3 dimension,
	const uint numberOfParticles,
	SimulationData simData)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= numberOfParticles)
		return;

	float3 particlePosition = make_float3(positions[index]);
	float3 particleVelocity = make_float3(velocity[index]);
	float4 particlePressure = calculatePressure(positions, velocity, index, particlePosition, particleVelocity, numberOfParticles, simData);

	particleVelocity += (gravity + make_float3(particlePressure)) * dt;

	particlePosition -= position;

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

	positionOut[index] = make_float4(particlePosition + position);
	velocity[index] = make_float4(particleVelocity);
}

__device__ float4 calculatePressure(const float4* __restrict__ position, const float4* __restrict__ velocity, uint index, float3 pos, float3 vel, uint numberOfParticles, SimulationData simData)
{
	float4 pressureVec = make_float4(0.f);
	float4 viscosityVec = pressureVec;
	for (uint i = 0; i < numberOfParticles; i++)
	{
		if (index == i)
			continue;

		float3 dirVec = pos - make_float3(position[i]);
		float dist = length(dirVec);//TODO: maybe use half_length

		if (dist > simData.interactionRadius * 1.0f || dist < 0.00001f)
			continue;

		float3 dirVecN = normalize(dirVec);
		float moveDir = dot(vel - make_float3(velocity[i]), dirVecN);
		float distRel = dist / simData.interactionRadius;

		// viscosity
		if (moveDir > 0)
		{
			float3 impulse = (1.f - distRel) * (simData.spring * moveDir + simData.springNear * moveDir * moveDir) * dirVecN;
			viscosityVec -= make_float4(impulse * 0.5f);//goes back to the caller-particle
											   //viscosityVec.w = 666.0f;
		}
		// *** v

		float oneminusx = 1.f - distRel;
		float sqx = oneminusx * oneminusx;
		float pressure = 1.f - simData.rho0 * (sqx * oneminusx - sqx);
		//float pressure = 1.f - (dist / simData.interactionRadius);
		////float pressure = amplitude * exp(-dist / interactionRadius);

		pressureVec += make_float4(pressure * dirVecN);
		//// pressureVec += vec4(dirVec, 0.f);

		//pressureVec.w += pressure;

		//break;
	}

	return pressureVec + viscosityVec;
}

extern "C" void cudaUpdate(
	float4* positions,
	float4* positionOut,
	float4* velocity,
	const float dt,
	const float3 gravity,
	const float3 position,
	const float3 dimension,
	const uint numberOfParticles,
	SimulationData simData)
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

	particleUpdate<<< num, threads >>>(positions, positionOut, velocity, dt, gravity, dimension, position, numberOfParticles, simData);
}
