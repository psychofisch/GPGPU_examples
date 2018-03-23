#pragma once

#include <cuda_runtime.h>

#include <thrust\device_vector.h>
#include <thrust\host_vector.h>
#include <thrust\transform.h>
#include <thrust\transform_reduce.h>
#include <thrust\iterator\zip_iterator.h>
#include <thrust\copy.h>

#include <helper_math.h>

#include "CollisionDefinitions.h"

__host__ __device__ bool operator==(float3& lhs, float3& rhs);

namespace ThrustHelper
{
	struct ThrustData
	{
		thrust::device_vector<MinMaxData> minMaxBuffer;
		thrust::device_vector<int> collisions;
	};

	ThrustData* setup(uint numberOfParticles);

	struct CollisionFunctor : public thrust::binary_function < MinMaxData, MinMaxData, int > {
		MinMaxData minMax;
		int collision;

		CollisionFunctor(float3 pos_, float3 vel_, SimulationData simData_);
		__host__ __device__ float4 operator()(float4 outerPos, float4 outerVel);
	};

	void thrustUpdate(
		ThrustData& tdata,
		MinMaxData* minMaxBuffer,
		int* collisionBuffer,
		const uint amountOfCubes);
}
