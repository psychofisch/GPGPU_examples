#include "thrust.h"

void thrustVectorAdd(int * vecA, int * vecB, int * vecC, const int numberOfElements)
{
	thrust::device_vector<int>  thrustA(numberOfElements),
								thrustB(numberOfElements),
								thrustC(numberOfElements);

	thrust::copy(vecA, vecA + numberOfElements, thrustA.begin());
	thrust::copy(vecB, vecB + numberOfElements, thrustB.begin());

	thrust::transform(thrustA.begin(), thrustA.end(), thrustB.begin(), thrustC.begin(), VectorAddFunctor(numberOfElements));

	thrust::copy(thrustC.begin(), thrustC.end(), vecC);
}

VectorAddFunctor::VectorAddFunctor(int noe_)
	:numberOfElements(noe_)
{
}

__host__ __device__ int VectorAddFunctor::operator()(int A, int B)
{
	return A + B;
}
