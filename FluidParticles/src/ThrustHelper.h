#pragma once

#include <cuda_runtime.h>
#include <helper_math.h>

#include <thrust\device_vector.h>
#include <thrust\host_vector.h>
#include <thrust\transform.h>
#include <thrust\transform_reduce.h>
#include <thrust\iterator\zip_iterator.h>
#include <thrust\copy.h>

#include "ParticleDefinitions.h"
#include "ParticlesCuda.h"
#include "cudaHelper.h"

typedef thrust::tuple<float4, float4> posVel;

__host__ __device__ bool operator==(float3& lhs, float3& rhs);

namespace ThrustHelper
{
	struct ThrustParticleData
	{
		thrust::device_vector<float4> position;
		thrust::device_vector<float4> positionOut;
		thrust::device_vector<float4> velocity;
		thrust::device_vector<MinMaxDataCuda> colliders;
	};

	std::unique_ptr<ThrustParticleData> setup(uint numberOfParticles);

	struct SimulationFunctor : public thrust::unary_function < posVel, posVel > {
		float dt;
		float3 dimension;
		float3 gravity;
		SimulationData simData;
		const MinMaxDataCuda* colliders;
		uint numberOfColliders;

		SimulationFunctor(float dt_, float3 dim_, float3 g_, SimulationData simData_, const MinMaxDataCuda* colliders_, uint numberOfColliders_);
		__host__ __device__ posVel operator()(posVel outerPos);
	};

	void thrustParticleUpdate(
		ThrustParticleData& tdata,
		float4* position,
		float4* positionOut,
		float4* velocity,
		const MinMaxDataCuda* staticColliders,
		const float dt,
		const float3 gravity,
		const float3 dimension,
		const uint numberOfParticles,
		const uint numberOfColliders,
		SimulationData simData);

	//void thrustCopyColliders(ThrustParticleData& tdata, std::vector<MinMaxDataCuda>& collision);
}
