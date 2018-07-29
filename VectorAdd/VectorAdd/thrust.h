#pragma once

#include <cuda_runtime.h>

#include <thrust\device_vector.h>
#include <thrust\host_vector.h>
#include <thrust\transform.h>
#include <thrust\iterator\zip_iterator.h>
#include <thrust\copy.h>

struct VectorAddFunctor : public thrust::binary_function < int, int, int > {
	const int numberOfElements;

	VectorAddFunctor(int aoc_);
	__host__ __device__ int operator()(int A, int B);
};

// this function is called from outside to call the Thrust implementation
void thrustVectorAdd(
	int* vecA,
	int* vecB,
	int* vecC,
	const int numberOfElements);
