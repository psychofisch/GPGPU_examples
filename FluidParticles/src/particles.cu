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

#include "cudaHelper.h"
#include "ParticleDefinitions.h"

//inline __device__ bool operator==(float3& lhs, float3& rhs)
//{
//	if (lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z)
//		return true;
//	else
//		return false;
//}

__device__ float dim(const float3& v, size_t i)
{
	switch (i)
	{
	case 0: return v.x;//no breaks required
	case 1:	return v.y;
	case 2:	return v.z;
	}

	return NAN;
}

__device__ inline float& dim(float3& v, size_t i)
{
	switch (i)
	{
	case 0: return v.x;//no breaks required
	case 1:	return v.y;
	case 2:	return v.z;
	}

	return v.x;//TODO: return something that indicates an error
}

__device__ float dim(const float4& v, size_t i)
{
	switch (i)
	{
	case 0: return v.x;//no breaks required
	case 1:	return v.y;
	case 2:	return v.z;
	case 3: return v.w;
	}

	return NAN;
}

__device__ float3 calculatePressure(const float4* __restrict__ positions, const float4* __restrict__ velocity, uint index, float3 pos, float3 vel, uint numberOfParticles, SimulationData simData);
__device__ bool ClipLine(int d, const MinMaxDataCuda aabbBox, const float3 v0, const float3 v1, float& f_low, float& f_high);
__device__ bool LineAABBIntersection(const MinMaxDataCuda aabbBox, const float3 v0, const float3 v1, float3& vecIntersection, float& flFraction);

__global__ void particleUpdate(
	const float4* __restrict__ positions,
	float4* __restrict__ positionOut,
	float4* __restrict__ velocity,
	const MinMaxDataCuda* staticColliders,
	const float dt, 
	const float3 gravity,
	const float3 position,
	const float3 dimension,
	const uint numberOfParticles,
	const uint numberOfColliders,
	SimulationData simData)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= numberOfParticles)
		return;

	float fluidDamp = 0.0;
	float particleSize = simData.interactionRadius * 0.1f;
	float3 worldAABBmin = make_float3(particleSize);
	float3 worldAABBmax = dimension - particleSize;

	float3 particlePosition = make_float3(positions[index]);
	float3 particleVelocity = make_float3(velocity[index]);
	float3 particlePressure = calculatePressure(positions, velocity, index, particlePosition, particleVelocity, numberOfParticles, simData);

	particlePosition -= position;

	// gravity
	particleVelocity += (gravity + particlePressure) * dt;
	// *** g

	float3 deltaVelocity = particleVelocity * dt;
	float3 sizeOffset = normalize(particleVelocity) * particleSize;
	float3 newPos = particlePosition + deltaVelocity;

	// static collision
	int collisionCnt = 3; //support multiple collisions
	for (int i = 0; i < numberOfColliders && collisionCnt > 0; i++)
	{
		MinMaxDataCuda currentAABB = staticColliders[i];
		float3 intersection;
		float fraction;
		bool result = false;

		result = LineAABBIntersection(currentAABB, particlePosition, newPos + sizeOffset, intersection, fraction);

		if (result == false)
			continue;

		if (intersection.x == currentAABB.max.x || intersection.x == currentAABB.min.x)
			particleVelocity.x *= -1.0;
		else if (intersection.y == currentAABB.max.y || intersection.y == currentAABB.min.y)
			particleVelocity.y *= -1.0;
		else if (intersection.z == currentAABB.max.z || intersection.z == currentAABB.min.z)
			particleVelocity.z *= -1.0;
		//else
		//	std::cout << "W00T!?\n";//DEBUG

		//particlePosition = intersection;
		newPos = intersection;
		break;// DEBUG! this prevents multiple collisions!

			  //	//ofVec3f reflection;
			  //	ofVec3f n = Particle::directions[closest];

			  //	// source -> https://math.stackexchange.com/questions/13261/how-to-get-a-reflection-vector#13266
			  //	particleVelocity = particleVelocity - (2 * particleVelocity.dot(n) * n);
			  //	particleVelocity *= fluidDamp;
			  //	
			  //	collisionCnt = 0;
			  //	//result = j;
			  //	//break;// OPT: do not delete this (30% performance loss)
	}
	// *** sc

	// bounding box collision
	float3 tmpVel = particleVelocity;
	for (int i = 0; i < 3; ++i)
	{
		if ((dim(newPos, i) > dim(worldAABBmax, i) && dim(tmpVel, i) > 0.0) // max boundary
			|| (dim(newPos, i) < dim(worldAABBmin, i) && dim(tmpVel, i) < 0.0) // min boundary
			)
		{
			/*if (newPos[i] < worldAABB.min[i])
			newPos[i] = worldAABB.min[i];
			else
			newPos[i] = worldAABB.max[i];*/

			dim(tmpVel, i) *= -fluidDamp;
		}
	}

	particleVelocity = tmpVel;
	// *** bbc

	// particleVelocity += dt * particleVelocity * -0.01f;//damping
	particlePosition += particleVelocity * dt;

	positionOut[index] = make_float4(particlePosition + position, 0.0f);
	velocity[index] = make_float4(particleVelocity, 0.0f);
}

__device__ float3 calculatePressure(const float4* __restrict__ position, const float4* __restrict__ velocity, uint index, float3 pos, float3 vel, uint numberOfParticles, SimulationData simData)
{
	float3 pressureVec = make_float3(0.f);
	float3 viscosityVec = pressureVec;
	float influence = 0.f;

	for (uint i = 0; i < numberOfParticles; i++)
	{
		if (index == i)
			continue;

		float3 dirVec = pos - make_float3(position[i]);
		float dist = length(dirVec);//TODO: maybe use half_length

		if (dist > simData.interactionRadius)
			continue;

		float3 dirVecN = normalize(dirVec);
		float moveDir = dot(vel - make_float3(velocity[i]), dirVecN);
		float distRel = 1.0f - dist / simData.interactionRadius;

		float sqx = distRel * distRel;

		influence += 1.0f;

		// viscosity
		if (true || moveDir > 0)
		{
			float factor = sqx * (simData.viscosity * moveDir);
			float3 impulse = factor * dirVecN;
			viscosityVec -= impulse;
		}
		// *** v

		float pressure = sqx * simData.pressureMultiplier;

		pressureVec += (pressure - simData.restPressure) * dirVecN;
	}

	//compress viscosity TODO: fix the root of this problem and not just limit it manually
	//float threshold = 50.0;
	if (influence > 0.f)
	{
		viscosityVec = viscosityVec / influence;
	}

	if (length(viscosityVec) > 100.0)
		viscosityVec = normalize(viscosityVec) * 100.0;
	//*** lv

	return pressureVec + viscosityVec;
}

__device__ bool ClipLine(int d, const MinMaxDataCuda aabbBox, const float3 v0, const float3 v1, float& f_low, float& f_high)
{
	// f_low and f_high are the results from all clipping so far. We'll write our results back out to those parameters.

	// f_dim_low and f_dim_high are the results we're calculating for this current dimension.
	float f_dim_low, f_dim_high;

	// Find the point of intersection in this dimension only as a fraction of the total vector http://youtu.be/USjbg5QXk3g?t=3m12s
	f_dim_low = (dim(aabbBox.min, d) - dim(v0, d)) / (dim(v1, d) - dim(v0, d));
	f_dim_high = (dim(aabbBox.max, d) - dim(v0, d)) / (dim(v1, d) - dim(v0, d));

	// Make sure low is less than high
	if (f_dim_high < f_dim_low)
	{
		float tmp = f_dim_high;
		f_dim_high = f_dim_low;
		f_dim_low = tmp;
	}

	// If this dimension's high is less than the low we got then we definitely missed. http://youtu.be/USjbg5QXk3g?t=7m16s
	if (f_dim_high < f_low)
		return false;

	// Likewise if the low is less than the high.
	if (f_dim_low > f_high)
		return false;

	// Add the clip from this dimension to the previous results http://youtu.be/USjbg5QXk3g?t=5m32s
	f_low = max(f_dim_low, f_low);
	f_high = min(f_dim_high, f_high);

	if (f_low > f_high)
		return false;

	return true;
}

// Find the intersection of a line from v0 to v1 and an axis-aligned bounding box http://www.youtube.com/watch?v=USjbg5QXk3g
__device__ bool LineAABBIntersection(const MinMaxDataCuda aabbBox, const float3 v0, const float3 v1, float3& vecIntersection, float& flFraction)
{
	float f_low = 0;
	float f_high = 1;

	if (!ClipLine(0, aabbBox, v0, v1, f_low, f_high))
		return false;

	if (!ClipLine(1, aabbBox, v0, v1, f_low, f_high))
		return false;

	if (!ClipLine(2, aabbBox, v0, v1, f_low, f_high))
		return false;

	// The formula for I: http://youtu.be/USjbg5QXk3g?t=6m24s
	float3 b = v1 - v0;
	vecIntersection = v0 + b * f_low;

	flFraction = f_low;

	return true;
}

extern "C" void cudaParticleUpdate(
	float4* positions,
	float4* positionOut,
	float4* velocity,
	MinMaxDataCuda* staticColliders,
	const float dt,
	const float3 gravity,
	const float3 position,
	const float3 dimension,
	const size_t numberOfParticles,
	const size_t numberOfColliders,
	SimulationData simData)
{
	cudaDeviceProp devProp;
	int device;
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&devProp, device);

	size_t num = 1;
	size_t threads = numberOfParticles;
	size_t maxThreads = devProp.maxThreadsPerBlock;

	if (numberOfParticles > maxThreads)
	{
		num = (size_t)ceilf(float(numberOfParticles) / maxThreads);
		threads = maxThreads;
	}

	particleUpdate <<< num, threads >>> (positions, positionOut, velocity, staticColliders, dt, gravity, dimension, position, numberOfParticles, numberOfColliders, simData);
}
