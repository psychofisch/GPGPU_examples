#include <cuda_runtime.h>

__global__ void cudaVectorAddKernel(const int *vectorA, const int *vectorB, int *vectorC, int numElements)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < numElements)
	{
		vectorC[i] = vectorA[i] + vectorB[i];
	}
}

extern "C" void cudaVectorAdd(const int *vectorA, const int *vectorB, int *vectorC, int numElements)
{
	int *cudaA, *cudaB, *cudaC;
	cudaMalloc((void**)&cudaA, sizeof(int) *  numElements);
	cudaMalloc((void**)&cudaB, sizeof(int) *  numElements);
	cudaMalloc((void**)&cudaC, sizeof(int) *  numElements);

	cudaMemcpy(cudaA, vectorA, sizeof(int) *  numElements, cudaMemcpyDefault);
	cudaMemcpy(cudaB, vectorB, sizeof(int) * numElements, cudaMemcpyDefault);

	int threads = 256;
	int blocks = (numElements + threads - 1) / threads;

	cudaVectorAddKernel <<< threads, blocks >>> (cudaA, cudaB, cudaC, numElements);

	cudaMemcpy(vectorC, cudaC, sizeof(int) *  numElements, cudaMemcpyDefault);

	cudaFree((void**)&cudaA);
	cudaFree((void**)&cudaB);
	cudaFree((void**)&cudaC);
}
