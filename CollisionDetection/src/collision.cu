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

__global__ void getCollisions(
	const float4* minMaxBuffer,
	int* collisionBuffer,
	const int amountOfCubes)
{
	const int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= amountOfCubes)
		return;

	float4 currentMin = minMaxBuffer[(index * 2)]; //min
	float4 currentMax = minMaxBuffer[(index * 2) + 1]; //max
	int result = -1;

	for (int j = 0; j < amountOfCubes; j++)
	{
		if (index == j)
			continue;
		//int cnt = 0;
		float4 otherMin = minMaxBuffer[j * 2];
		float4 otherMax = minMaxBuffer[(j * 2) + 1];

		if (((  otherMin.x < currentMax.x && otherMin.x > currentMin.x)
			|| (otherMax.x < currentMax.x && otherMax.x > currentMin.x)
			|| (otherMax.x > currentMax.x && otherMin.x < currentMin.x)
			|| (otherMax.x < currentMax.x && otherMin.x > currentMin.x))
			&&
			((  otherMin.z < currentMax.z && otherMin.z > currentMin.z)
			|| (otherMax.z < currentMax.z && otherMax.z > currentMin.z)
			|| (otherMax.z > currentMax.z && otherMin.z < currentMin.z)
			|| (otherMax.z < currentMax.z && otherMin.z > currentMin.z))
			&&	
			((	otherMin.y < currentMax.y && otherMin.y > currentMin.y)
			|| (otherMax.y < currentMax.y && otherMax.y > currentMin.y)
			|| (otherMax.y > currentMax.y && otherMin.y < currentMin.y)
			|| (otherMax.y < currentMax.y && otherMin.y > currentMin.y))
			) // TODO: optimize this
		{
			result = j;
			break;// OPT: do not delete this (30% performance loss)
		}
	}

	collisionBuffer[index] = result;
}

extern "C" void cudaGetCollisions(
	float4* minMaxBuffer,
	int* collisionBuffer,
	const int amountOfCubes)
{
	cudaDeviceProp devProp;
	int device;
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&devProp, device);

	int num = 1;
	int threads = amountOfCubes;
	int maxThreads = devProp.maxThreadsPerBlock;

	if (amountOfCubes > maxThreads)
	{
		num = (int)ceilf(float(amountOfCubes) / maxThreads);
		threads = maxThreads;
	}

	getCollisions <<< num, threads >>>(minMaxBuffer, collisionBuffer, amountOfCubes);
}
