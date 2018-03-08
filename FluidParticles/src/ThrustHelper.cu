#include "ThrustHelper.h"

inline __host__ __device__ bool operator==(float3& lhs, float3& rhs)
{
	if (lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z)
		return true;
	else
		return false;
}

ThrustHelper::InvertFunctor::InvertFunctor(float4 pos_, float4 vel_, SimulationData simData_)
	:pos(pos_),
	vel(vel_),
	simData(simData_)
{}

__host__ __device__ float4 ThrustHelper::InvertFunctor::operator()(float4 outerPos, float4 outerVel)
{
	float3 result = make_float3(0.f);
	float3 pressureVec, viscosityVec;

	//if (outerPos == pos)
	//	return result;

	//float3 dirVec = pos - outerPos;
	//float dist = length(dirVec);//TODO: maybe use half_length

	//if (dist > simData.interactionRadius * 1.0f || dist < 0.00001f)
	//	return result;

	//float3 dirVecN = normalize(dirVec);
	//float moveDir = dot(vel - outerVel, dirVecN);
	//float distRel = dist / simData.interactionRadius;

	//// viscosity
	//if (moveDir > 0)
	//{
	//	float3 impulse = (1.f - distRel) * (simData.spring * moveDir + simData.springNear * moveDir * moveDir) * dirVecN;
	//	viscosityVec = (impulse * 0.5f);//goes back to the caller-particle
	//									//viscosityVec.w = 666.0f;
	//}
	//// *** v

	//float oneminusx = 1.f - distRel;
	//float sqx = oneminusx * oneminusx;
	//float pressure = 1.f - simData.rho0 * (sqx * oneminusx - sqx);

	//pressureVec = pressure * dirVecN;

	//break;

	//return make_float4(pressureVec + viscosityVec);
	return make_float4(0.5f);
}

void ThrustHelper::thrustUpdate(
	thrust::device_vector<float4>& position,
	thrust::device_vector<float4>& positionOut,
	thrust::device_vector<float4>& velocity,
	const float dt,
	const float3 gravity,
	const float3 dimension,
	const uint numberOfParticles,
	SimulationData simData)
{
	/*thrust::device_vector<float4> devicePos = position;
	thrust::device_vector<float4> deviceVel = velocity;
	thrust::device_vector<float4> deviceOut = positionOut;*/

	float4 tmpPos = position[0];
	float4 tmpVel = velocity[0];

	//thrust::transform(position.begin(), position.end(), velocity.begin(), positionOut.begin(), InvertFunctor(tmpPos, tmpVel, simData));

	thrust::fill(positionOut.begin(), positionOut.end(), make_float4(0.5f));

	//thrust::copy(deviceOut.begin(), deviceOut.end(), positionOut.begin());
}
