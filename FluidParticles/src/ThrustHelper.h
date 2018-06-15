#pragma once

#include "ParticleDefinitions.h"

#include <cuda_runtime.h>

#include <thrust\device_vector.h>
#include <thrust\host_vector.h>
#include <thrust\transform.h>
#include <thrust\transform_reduce.h>
#include <thrust\iterator\zip_iterator.h>
#include <thrust\copy.h>

#include <helper_math.h>

__host__ __device__ bool operator==(float3& lhs, float3& rhs);

namespace ThrustHelper
{
	struct ThrustParticleData
	{
		thrust::device_vector<float4> position;
		thrust::device_vector<float4> positionOut;
		thrust::device_vector<float4> velocity;
	};

	ThrustParticleData* setup(uint numberOfParticles);

	struct PressureFunctor : public thrust::binary_function < float4, float4, float4 > {
		float3 pos;
		float3 vel;
		SimulationData simData;

		PressureFunctor(float3 pos_, float3 vel_, SimulationData simData_);
		__host__ __device__ float4 operator()(float4 outerPos, float4 outerVel);
	};

	struct SimulationFunctor : public thrust::binary_function < float4, float4, float4 > {
		float dt;
		float3 dimension;
		float3 gravity;
		SimulationData simData;

		SimulationFunctor(float dt_, float3 dim_, float3 g_, SimulationData simData_);
		__host__ __device__ float4 operator()(float4 outerPos, float4 outerVel);
	};

	void thrustParticleUpdate(
		ThrustParticleData& tdata,
		float4* position,
		float4* positionOut,
		float4* velocity,
		const float dt,
		const float3 gravity,
		const float3 dimension,
		const uint numberOfParticles,
		SimulationData simData);
}
