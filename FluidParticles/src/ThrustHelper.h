#pragma once

#include "ParticleDefinitions.h"

#include <cuda_runtime.h>

#include <thrust\device_vector.h>
#include <thrust\host_vector.h>
#include <thrust\transform.h>
#include <thrust\iterator\zip_iterator.h>
#include <thrust\copy.h>

#include <helper_math.h>

__host__ __device__ bool operator==(float3& lhs, float3& rhs);

namespace ThrustHelper
{
	struct InvertFunctor : public thrust::binary_function < float4, float4, float4 > {
		float4 pos;
		float4 vel;
		SimulationData simData;

		InvertFunctor(float4 pos_, float4 vel_, SimulationData simData_);
		__host__ __device__ float4 operator()(float4 outerPos, float4 outerVel);
	};

	void thrustUpdate(
		thrust::host_vector<float4>& position,
		thrust::host_vector<float4>& positionOut,
		thrust::host_vector<float4>& velocity,
		const float dt,
		const float3 gravity,
		const float3 dimension,
		const uint numberOfParticles,
		SimulationData simData);
}
