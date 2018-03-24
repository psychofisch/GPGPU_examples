#include "ThrustHelper.h"

ThrustHelper::CollisionFunctor::CollisionFunctor(uint td_, MinMaxDataThrust* mmd_)
	:tdata(td_),
	minMaxRaw(mmd_)
{}

__host__ __device__ int ThrustHelper::CollisionFunctor::operator()(thrustMinMaxTuple minMax)
{
	int result = -1;
	float3 currentMin = make_float3(thrust::get<0>(minMax).min);
	float3 currentMax = make_float3(thrust::get<0>(minMax).max);
	int i = thrust::get<1>(minMax);
	for (int j = 0; j < tdata; j++)
	{
		if (i == j)
			continue;
		//int cnt = 0;
		float3 otherMin = make_float3(minMaxRaw[j].min);
		float3 otherMax = make_float3(minMaxRaw[j].max);

		bool loop = true;
		int p = 0;
		if (((otherMin.x < currentMax.x && otherMin.x > currentMin.x)
			|| (otherMax.x < currentMax.x && otherMax.x > currentMin.x)
			|| (otherMax.x > currentMax.x && otherMin.x < currentMin.x)
			|| (otherMax.x < currentMax.x && otherMin.x > currentMin.x))
			&&
			((otherMin.z < currentMax.z && otherMin.z > currentMin.z)
				|| (otherMax.z < currentMax.z && otherMax.z > currentMin.z)
				|| (otherMax.z > currentMax.z && otherMin.z < currentMin.z)
				|| (otherMax.z < currentMax.z && otherMin.z > currentMin.z))
			&&
			((otherMin.y < currentMax.y && otherMin.y > currentMin.y)
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
	/*thrust::device_vector<float4> devicePos(position, position + numberOfParticles);
	thrust::device_vector<float4> deviceVel(velocity, velocity + numberOfParticles);
	thrust::device_vector<float4> deviceOut(numberOfParticles);*/

	if (tdata.collisions.size() < amountOfCubes)
	{
		std::cout << "Thrust: allocating memory for " << amountOfCubes << " cubes.\n";

		tdata.minMaxBuffer.assign(minMaxBuffer, minMaxBuffer + amountOfCubes);
		tdata.collisions.resize(amountOfCubes);
	}
	
	MinMaxDataThrust* minMaxDevice = thrust::raw_pointer_cast(tdata.minMaxBuffer.data());

	// calculate simulation
	thrust::transform(
		thrust::make_zip_iterator(make_tuple(tdata.minMaxBuffer.begin(), tdata.collisions.begin())),
		thrust::make_zip_iterator(make_tuple(tdata.minMaxBuffer.end(), tdata.collisions.end())),
		tdata.collisions.begin(),
		CollisionFunctor(amountOfCubes, minMaxDevice)
	);
	
	thrust::copy(tdata.collisions.begin(), tdata.collisions.end(), collisionBuffer);
}
