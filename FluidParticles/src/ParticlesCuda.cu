#include "ParticlesCuda.h"

__global__ void particleUpdate(
	const float4* __restrict__ positions,
	float4* __restrict__ positionOut,
	float4* __restrict__ velocity,
	const MinMaxDataCuda* staticColliders,
	const float dt, 
	const float3 gravity,
	const float3 position,
	const float3 dimension,
	const size_t numberOfParticles,
	const size_t numberOfColliders,
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
	for (int i = 0; i < numberOfColliders; i++)
	{
		MinMaxDataCuda currentAABB = staticColliders[i];
		float3 intersection;
		float fraction;
		bool result = false;

		result = LineAABBIntersection(currentAABB, particlePosition, newPos + sizeOffset, intersection, fraction);

		if (result == false)
			continue;

		if (intersection.x == currentAABB.max.x || intersection.x == currentAABB.min.x)
			particleVelocity.x *= -fluidDamp;
		else if (intersection.y == currentAABB.max.y || intersection.y == currentAABB.min.y)
			particleVelocity.y *= -fluidDamp;
		else if (intersection.z == currentAABB.max.z || intersection.z == currentAABB.min.z)
			particleVelocity.z *= -fluidDamp;

		newPos = intersection;
		break;
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
			dim(tmpVel, i) *= -fluidDamp;
		}
	}

	particleVelocity = tmpVel;
	// *** bbc

	particlePosition += particleVelocity * dt;

	positionOut[index] = make_float4(particlePosition + position, length(particleVelocity));
	velocity[index] = make_float4(particleVelocity, 0.0f);
}

__device__ __host__ float3 calculatePressure(const float4* __restrict__ position, const float4* __restrict__ velocity, uint index, float3 pos, float3 vel, uint numberOfParticles, SimulationData simData)
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
	if (influence > 0.f)
	{
		viscosityVec = viscosityVec / influence;
	}

	if (length(viscosityVec) > 100.0)
		viscosityVec = normalize(viscosityVec) * 100.0;
	//*** lv

	return pressureVec + viscosityVec;
}

void cudaParticleUpdate(
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
