#include "ThrustHelper.h"

ThrustHelper::CollisionFunctor::CollisionFunctor(uint td_, MinMaxDataThrust* mmd_)
	:tdata(td_),
	minMaxRaw(mmd_)
{}

__host__ __device__ int ThrustHelper::CollisionFunctor::operator()(MinMaxDataThrust minMax)
{
	int result = -666;
	float3 currentMin = minMax.min;
	float3 currentMax = minMax.max;
	int i = minMax.id;
	for (int j = 0; j < tdata; j++)
	{
		if (i == j)
			continue;
		//int cnt = 0;
		float3 otherMin = minMaxRaw[j].min;
		float3 otherMax = minMaxRaw[j].max;

		bool loop = true;
		int p = 0;
		if (((  otherMin.x < currentMax.x && otherMin.x > currentMin.x)
			|| (otherMax.x < currentMax.x && otherMax.x > currentMin.x)
			|| (otherMax.x > currentMax.x && otherMin.x < currentMin.x)
			|| (otherMax.x < currentMax.x && otherMin.x > currentMin.x))
			&&
		((		otherMin.z < currentMax.z && otherMin.z > currentMin.z)
			|| (otherMax.z < currentMax.z && otherMax.z > currentMin.z)
			|| (otherMax.z > currentMax.z && otherMin.z < currentMin.z)
			|| (otherMax.z < currentMax.z && otherMin.z > currentMin.z))
			&&
		((		otherMin.y < currentMax.y && otherMin.y > currentMin.y)
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
		tdata.collisions.resize(amountOfCubes);
	}
	
	tdata.minMaxBuffer.assign(minMaxBuffer, minMaxBuffer + amountOfCubes);

	MinMaxDataThrust* minMaxDevice = thrust::raw_pointer_cast(tdata.minMaxBuffer.data());
	// calculate simulation
	thrust::transform(
		tdata.minMaxBuffer.begin(),
		tdata.minMaxBuffer.end(),
		tdata.collisions.begin(),
		CollisionFunctor(amountOfCubes, minMaxDevice)
	);
	
	thrust::copy(tdata.collisions.begin(), tdata.collisions.end(), collisionBuffer);
}
