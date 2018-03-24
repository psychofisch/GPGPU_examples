#pragma once

#include <cuda_runtime.h>

#include <thrust\device_vector.h>
#include <thrust\host_vector.h>
#include <thrust\transform.h>
#include <thrust\iterator\zip_iterator.h>
#include <thrust\copy.h>

#include <helper_math.h>

#include "CollisionDefinitions.h"

typedef thrust::tuple<MinMaxData, int> thrustMinMax;

namespace ThrustHelper
{
	struct ThrustData
	{
		thrust::device_vector<MinMaxData> minMaxBuffer;
		thrust::device_vector<int> collisions;
	};

	struct CollisionFunctor : public thrust::unary_function < thrustMinMax, int > {
		ThrustData& tdata;
		MinMaxData* minMaxRaw;

		CollisionFunctor(ThrustData& td_, MinMaxData* mmd_);
		__host__ __device__ int operator()(thrustMinMax minMax);
	};

	void thrustGetCollisions(
		ThrustData& tdata,
		MinMaxData* minMaxBuffer,
		int* collisionBuffer,
		const uint amountOfCubes);
}
