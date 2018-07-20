#pragma once

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

//__device__ __host__ float dim(const float3& v, size_t i);
//__device__ __host__ float& dim(float3& v, size_t i);
//__device__ __host__ float dim(const float4& v, size_t i);
//__device__ __host__ bool ClipLine(int d, const MinMaxDataCuda aabbBox, const float3 v0, const float3 v1, float& f_low, float& f_high);
//__device__ __host__ bool LineAABBIntersection(const MinMaxDataCuda aabbBox, const float3 v0, const float3 v1, float3& vecIntersection, float& flFraction);

__device__ __host__ float3 calculatePressure(const float4* __restrict__ positions, const float4* __restrict__ velocity, uint index, float3 pos, float3 vel, uint numberOfParticles, SimulationData simData);

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
	SimulationData simData);

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
	SimulationData simData);
