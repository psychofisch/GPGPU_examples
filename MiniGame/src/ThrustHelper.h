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

struct MinMaxDataThrust
{
	float3 min, max;
	int id;
};

namespace ThrustHelper
{
	// storage of the device vectors
	struct ThrustCollisionData
	{
		thrust::device_vector<MinMaxDataThrust> minMaxBuffer;
		thrust::device_vector<int> collisions;
	};

	struct ThrustParticleData
	{
		thrust::device_vector<float4> position;
		thrust::device_vector<float4> positionOut;
		thrust::device_vector<float4> velocity;
	};

	// init function
	ThrustParticleData* setup(uint numberOfParticles);

	// structs to hold the functions that are used by Thrust
	struct CollisionFunctor : public thrust::unary_function < MinMaxDataThrust, int > {
		uint amountOfCubes;
		MinMaxDataThrust* minMaxRaw;

		CollisionFunctor(uint aoc_, MinMaxDataThrust* mmd_);
		__host__ __device__ int operator()(MinMaxDataThrust minMax);
	};

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

	// functions that are used from "outside"
	void thrustGetCollisions(
		ThrustCollisionData& tdata,
		MinMaxDataThrust* minMaxBuffer,
		int* collisionBuffer,
		const uint amountOfCubes);

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
