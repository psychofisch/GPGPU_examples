#include "ThrustHelper.h"

namespace ThrustHelper
{
	inline __host__ __device__ bool operator==(float3& lhs, float3& rhs)
	{
		if (lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z)
			return true;
		else
			return false;
	}

	struct InvertFunctor : thrust::binary_function < float3, float3, float3> {
		float3 pos;
		float3 vel;
		SimulationData simData;

		InvertFunctor(float3 pos_, float3 vel_, SimulationData simData_)
			:pos(pos_),
			vel(vel_),
			simData(simData_)
		{}

		__host__ __device__ float3 operator()(float3 outerPos, float3 outerVel)
		{
			float3 result = make_float3(0.f);
			float3 pressureVec, viscosityVec;

			if(outerPos == pos)
				return result;

			float3 dirVec = pos - outerPos;
			float dist = length(dirVec);//TODO: maybe use half_length

			if (dist > simData.interactionRadius * 1.0f || dist < 0.00001f)
				return result;

			float3 dirVecN = normalize(dirVec);
			float moveDir = dot(vel - outerVel, dirVecN);
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
			//float pressure = 1.f - (dist / simData.interactionRadius);
			////float pressure = amplitude * exp(-dist / interactionRadius);

			pressureVec = pressure * dirVecN;
			//// pressureVec += vec4(dirVec, 0.f);

			//pressureVec.w += pressure;

			//break;

			return pressureVec + viscosityVec;
		}
	};

	void thrustUpdate(
		float4* position,
		float4* positionOut,
		float4* velocity,
		const float dt,
		const float3 gravity,
		const float3 dimension,
		const uint numberOfParticles,
		SimulationData simData)
	{
	}
}
