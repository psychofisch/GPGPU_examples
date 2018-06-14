#include "ThrustHelper.h"

ThrustHelper::CollisionFunctor::CollisionFunctor(uint aoc_, MinMaxDataThrust* mmd_)
	:amountOfCubes(aoc_),
	minMaxRaw(mmd_)
{}

__host__ __device__ int ThrustHelper::CollisionFunctor::operator()(MinMaxDataThrust minMax, int index)
{
	int result = -1;
	float3 currentMin = make_float3(minMax.min);
	float3 currentMax = make_float3(minMax.max);
	int i = index;

	for (uint j = 0; j < amountOfCubes; j++)
	{
		if (i == j)
			continue;

		float3 otherMin = make_float3(minMaxRaw[j].min);
		float3 otherMax = make_float3(minMaxRaw[j].max);

		if (((  otherMin.x < currentMax.x && otherMin.x > currentMin.x)
			|| (otherMax.x < currentMax.x && otherMax.x > currentMin.x)
			|| (otherMax.x > currentMax.x && otherMin.x < currentMin.x)
			|| (otherMax.x < currentMax.x && otherMin.x > currentMin.x))
			&&
			((	otherMin.z < currentMax.z && otherMin.z > currentMin.z)
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

	return result;
}

void ThrustHelper::thrustGetCollisions(
	ThrustData& tdata,
	MinMaxDataThrust* minMaxBuffer,
	int* collisionBuffer,
	const uint amountOfCubes)
{
	if (tdata.collisions.size() < amountOfCubes)
	{
		std::cout << "Thrust: allocating memory for " << amountOfCubes << " cubes.\n";
		tdata.collisions.resize(amountOfCubes);
	}
	
	// copy the minMaxBuffer to the device
	tdata.minMaxBuffer.assign(minMaxBuffer, minMaxBuffer + amountOfCubes);

	MinMaxDataThrust* minMaxDevice = thrust::raw_pointer_cast(tdata.minMaxBuffer.data());
	
	// calculate simulation with Thrust
	/*thrust::transform(
		tdata.minMaxBuffer.begin(),
		tdata.minMaxBuffer.end(),
		tdata.collisions.begin(),
		CollisionFunctor(amountOfCubes, minMaxDevice)
	);*/
	
	thrust::transform(
		tdata.minMaxBuffer.begin(),
		tdata.minMaxBuffer.end(),
		thrust::make_counting_iterator(0),
		tdata.collisions.begin(),
		CollisionFunctor(amountOfCubes, minMaxDevice)
	);

	// copy the device collision buffer to the host
	thrust::copy(tdata.collisions.begin(), tdata.collisions.end(), collisionBuffer);
}
