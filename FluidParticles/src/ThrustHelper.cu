#include "ThrustHelper.h"

inline __host__ __device__ bool operator==(float3& lhs, float3& rhs)
{
	if (lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z)
		return true;
	else
		return false;
}

ThrustHelper::PressureFunctor::PressureFunctor(float4 pos_, float4 vel_, SimulationData simData_)
	:pos(make_float3(pos_)),
	vel(make_float3(vel_)),
	simData(simData_)
{}

__host__ __device__ float4 ThrustHelper::PressureFunctor::operator()(float4 outerPos, float4 outerVel)
{
	float3 oPos = make_float3(outerPos);
	float3 oVel = make_float3(outerVel);

	float3 result = make_float3(0.f);
	float3 pressureVec, viscosityVec;

	if (oPos == pos)
		return make_float4(result);

	float3 dirVec = pos - oPos;
	float dist = length(dirVec);//TODO: maybe use half_length

	if (dist > simData.interactionRadius * 1.0f || dist < 0.00001f)
		return make_float4(result);

	float3 dirVecN = normalize(dirVec);
	float moveDir = dot(vel - oVel, dirVecN);
	float distRel = dist / simData.interactionRadius;

	// viscosity
	if (moveDir > 0)
	{
		float3 impulse = (1.f - distRel) * (simData.spring * moveDir + simData.springNear * moveDir * moveDir) * dirVecN;
		viscosityVec = (impulse * 0.5f);//goes back to the caller-particle
										//viscosityVec.w = 666.0f;
	}
	// *** v

	float oneminusx = 1.f - distRel;
	float sqx = oneminusx * oneminusx;
	float pressure = 1.f - simData.rho0 * (sqx * oneminusx - sqx);

	pressureVec = pressure * dirVecN;

	return make_float4(pressureVec + viscosityVec);
}

ThrustHelper::SimulationFunctor::SimulationFunctor(float dt_, SimulationData simData_)
	:dt(dt_),
	simData(simData_)
{}

__host__ __device__ float4 ThrustHelper::SimulationFunctor::operator()(float4 outerPos, float4 outerVel, float4 pressure)
{
	float3 oPos = make_float3(outerPos);
	float3 oVel = make_float3(outerVel);

	float3 result = make_float3(0.f);

	

	return make_float4(result);
}

void ThrustHelper::thrustUpdate(
	thrust::host_vector<float4>& position,
	thrust::host_vector<float4>& positionOut,
	thrust::host_vector<float4>& velocity,
	const float dt,
	const float3 gravity,
	const float3 dimension,
	const uint numberOfParticles,
	SimulationData simData)
{
	thrust::device_vector<float4> devicePos = position;
	thrust::device_vector<float4> deviceVel = velocity;
	thrust::device_vector<float4> deviceOut(numberOfParticles);

	float4 tmpPos = position[0];
	float4 tmpVel = velocity[0];

	// calculate pressure
	thrust::transform(devicePos.begin(), devicePos.end(), deviceVel.begin(), deviceOut.begin(), PressureFunctor(tmpPos, tmpVel, simData));

	// add pressure to velocity
	thrust::transform(deviceVel.begin(), deviceVel.end(),//input 1
		deviceOut.begin(),//input 2
		deviceVel.begin(),
		(thrust::placeholders::_1 + thrust::placeholders::_2));

	// calculate simulation
	//thrust::transform(devicePos.begin(), devicePos.end(), deviceVel.begin(), deviceOut.begin(), SimulationFunctor(tmpPos, tmpVel, simData));

	thrust::copy(deviceOut.begin(), deviceOut.end(), positionOut.begin());
}
