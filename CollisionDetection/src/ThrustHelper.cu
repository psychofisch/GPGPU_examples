#include "ThrustHelper.h"

ThrustHelper::CollisionFunctor::CollisionFunctor(ThrustData& td_, MinMaxData* mmd_)
	:tdata(td_),
	minMaxRaw(mmd_)
{}

__host__ __device__ int ThrustHelper::CollisionFunctor::operator()(thrustMinMax minMax)
{
	ofVec3f currentMin = thrust::get<0>(minMax).min;
	ofVec3f currentMax = thrust::get<0>(minMax).max;
	int i = thrust::get<1>(minMax);
	int result = -1;
	for (int j = 0; j < tdata.minMaxBuffer.size(); j++)
	{
		if (i == j)
			continue;
		//int cnt = 0;
		ofVec3f otherMin = minMaxRaw[j].min;
		ofVec3f otherMax = minMaxRaw[j].max;

		bool loop = true;
		int p = 0;
		while (loop && p <= 3)
		{
			if ((otherMin[p] < currentMax[p] && otherMin[p] > currentMin[p])
				|| (otherMax[p] < currentMax[p] && otherMax[p] > currentMin[p])
				|| (otherMax[p] > currentMax[p] && otherMin[p] < currentMin[p])
				|| (otherMax[p] < currentMax[p] && otherMin[p] > currentMin[p])) // TODO: optimize this
			{
				loop = true;
				++p;
			}
			else
			{
				loop = false;
			}
		}

		if (p >= 3)
		{
			result = j;
			break;
		}
	}

	return result;
}

void ThrustHelper::thrustGetCollisions(
	ThrustData& tdata,
	MinMaxData* minMaxBuffer,
	int* collisionBuffer,
	const uint amountOfCubes)
{
	/*thrust::device_vector<float4> devicePos(position, position + numberOfParticles);
	thrust::device_vector<float4> deviceVel(velocity, velocity + numberOfParticles);
	thrust::device_vector<float4> deviceOut(numberOfParticles);*/

	if (tdata.collisions.size() < amountOfCubes)
	{
		tdata.minMaxBuffer.assign(minMaxBuffer, minMaxBuffer + amountOfCubes);
		tdata.collisions.resize(amountOfCubes);
	}
	
	MinMaxData* minMaxDevice = thrust::raw_pointer_cast(tdata.minMaxBuffer.data());

	// calculate simulation
	thrust::transform(
		thrust::make_zip_iterator(make_tuple(tdata.minMaxBuffer.begin(), tdata.collisions.begin())),
		thrust::make_zip_iterator(make_tuple(tdata.minMaxBuffer.end(), tdata.collisions.end())),
		tdata.collisions.begin(),
		CollisionFunctor(tdata, minMaxDevice)
	);
	
	//thrust::copy(tdata.positionOut.begin(), tdata.positionOut.end(), positionOut);
}
