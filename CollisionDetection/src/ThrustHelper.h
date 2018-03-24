#pragma once

#include <cuda_runtime.h>

#include <thrust\device_vector.h>
#include <thrust\host_vector.h>
#include <thrust\transform.h>
#include <thrust\iterator\zip_iterator.h>
#include <thrust\copy.h>

#include <helper_math.h>

struct MinMaxDataThrust
{
	float3 min, max;
	int id;
};

//typedef thrust::tuple<MinMaxDataThrust, int> thrustMinMaxTuple;

namespace ThrustHelper
{
	struct ThrustData
	{
		thrust::device_vector<MinMaxDataThrust> minMaxBuffer;
		thrust::device_vector<int> collisions;
	};

	struct CollisionFunctor : public thrust::unary_function < MinMaxDataThrust, int > {
		uint tdata;
		MinMaxDataThrust* minMaxRaw;

		CollisionFunctor(uint td_, MinMaxDataThrust* mmd_);
		__host__ __device__ int operator()(MinMaxDataThrust minMax);
	};

	void thrustGetCollisions(
		ThrustData& tdata,
		MinMaxDataThrust* minMaxBuffer,
		int* collisionBuffer,
		const uint amountOfCubes);
}
