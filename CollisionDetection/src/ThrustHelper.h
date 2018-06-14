#pragma once

#include <cuda_runtime.h>

#include <thrust\device_vector.h>
#include <thrust\host_vector.h>
#include <thrust\transform.h>
#include <thrust\iterator\zip_iterator.h>
#include <thrust\copy.h>

#include <helper_math.h>

//typedef thrust::tuple<MinMaxDataThrust, int> thrustMinMaxTuple;

namespace ThrustHelper
{
	// storage of the device vectors
	struct MinMaxDataThrust
	{
		float4 min, max;
	};

	struct ThrustData
	{
		thrust::device_vector<MinMaxDataThrust> minMaxBuffer;
		thrust::device_vector<int> collisions;
	};

	// define a struct to hold the function that is used by Thrust
	struct CollisionFunctor : public thrust::binary_function < MinMaxDataThrust, int, int > {
		uint amountOfCubes;
		MinMaxDataThrust* minMaxRaw;

		CollisionFunctor(uint aoc_, MinMaxDataThrust* mmd_);
		__host__ __device__ int operator()(MinMaxDataThrust minMax, int index);
	};

	// this function is called from outside to call the Thrust implementation
	void thrustGetCollisions(
		ThrustData& tdata,
		MinMaxDataThrust* minMaxBuffer,
		int* collisionBuffer,
		const uint amountOfCubes);
}
