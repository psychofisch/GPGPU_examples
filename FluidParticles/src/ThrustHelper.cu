#include "ThrustHelper.h"

inline __host__ __device__ bool operator==(float3& lhs, float3& rhs)
{
	if (lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z)
		return true;
	else
		return false;
}

ThrustHelper::SimulationFunctor::SimulationFunctor(float dt_, float3 dim_, float3 g_, SimulationData simData_)
	:dt(dt_),
	dimension(dim_),
	gravity(g_),
	simData(simData_)
{}

__host__ __device__ float4 ThrustHelper::SimulationFunctor::operator()(float4 outerPos, float4 outerVel)
{
	float3 particlePosition = make_float3(outerPos);
	float3 particleVelocity = make_float3(outerVel);

	float3 result = make_float3(0.f);

	particleVelocity += gravity * dt;

	// static collision
	//TODO: write some kind of for-loop
	if ((particlePosition.x + particleVelocity.x * dt > dimension.x && particleVelocity.x > 0.f) || (particlePosition.x + particleVelocity.x * dt < 0.f && particleVelocity.x < 0.f))
	{
		if (particlePosition.x + particleVelocity.x * dt < 0.f)
			particlePosition.x = 0.f;
		else
			particlePosition.x = dimension.x;

		particleVelocity.x *= -.3f;
	}

	if ((particlePosition.y + particleVelocity.y * dt  > dimension.y && particleVelocity.y > 0.f) || (particlePosition.y + particleVelocity.y * dt < 0.f && particleVelocity.y < 0.f))
	{
		if (particlePosition.y + particleVelocity.y * dt < 0.f)
			particlePosition.y = 0.f;
		else
			particlePosition.y = dimension.y;

		particleVelocity.y *= -.3f;
	}

	if ((particlePosition.z + particleVelocity.z * dt > dimension.z && particleVelocity.z > 0.f) || (particlePosition.z + particleVelocity.z * dt < 0.f && particleVelocity.z < 0.f))
	{
		if (particlePosition.z + particleVelocity.z * dt < 0.f)
			particlePosition.z = 0.f;
		else
			particlePosition.z = dimension.z;

		particleVelocity.z *= -.3f;
	}
	// *** sc

	// particleVelocity += dt * particleVelocity * -0.01f;//damping
	particlePosition += particleVelocity * dt;

	//positionOut[index] = make_float4(particlePosition);

	return make_float4(particlePosition);
}

ThrustHelper::PressureFunctor::PressureFunctor(float3 pos_, float3 vel_, SimulationData simData_)
	:pos(pos_),
	vel(vel_),
	simData(simData_)
{}

__host__ __device__ float4 ThrustHelper::PressureFunctor::operator()(float4 outerPos, float4 outerVel)
{
	float3 pressureVec = make_float3(0.f);
	float3 viscosityVec = pressureVec;
	float influence = 0.f;

	float3 dirVec = pos - make_float3(outerPos);
	float dist = length(dirVec);//TODO: maybe use half_length

	if (dist > simData.interactionRadius || dist == 0.0f)
		return make_float4(0.f);

	float3 dirVecN = normalize(dirVec);
	float moveDir = dot(vel - make_float3(outerVel), dirVecN);
	float distRel = 1.0f - dist / simData.interactionRadius;

	float sqx = distRel * distRel;

	influence += 1.0f;

	// viscosity
	if (true || moveDir > 0)
	{
		float factor = sqx * (simData.viscosity * moveDir);
		float3 impulse = factor * dirVecN;
		viscosityVec -= impulse;
	}
	// *** v

	float pressure = sqx * simData.pressureMultiplier;

	pressureVec += (pressure - simData.restPressure) * dirVecN;

	//compress viscosity TODO: fix the root of this problem and not just limit it manually
	//float threshold = 50.0;
	if (influence > 0.f)
	{
		viscosityVec = viscosityVec / influence;
	}

	if (length(viscosityVec) > 100.0)
		viscosityVec = normalize(viscosityVec) * 100.0;
	//*** lv

	return make_float4(pressureVec + viscosityVec);
}

std::unique_ptr<ThrustHelper::ThrustParticleData> ThrustHelper::setup(uint numberOfParticles)
{
	auto r = std::make_unique<ThrustHelper::ThrustParticleData>();
	r->position.reserve(numberOfParticles);
	r->velocity.reserve(numberOfParticles);
	r->positionOut.reserve(numberOfParticles);
	return r;
}

void ThrustHelper::thrustParticleUpdate(
	ThrustParticleData& tdata,
	float4* position,
	float4* positionOut,
	float4* velocity,
	const float dt,
	const float3 gravity,
	const float3 dimension,
	const uint numberOfParticles,
	SimulationData simData)
{
	/*thrust::device_vector<float4> devicePos(position, position + numberOfParticles);
	thrust::device_vector<float4> deviceVel(velocity, velocity + numberOfParticles);
	thrust::device_vector<float4> deviceOut(numberOfParticles);*/
	tdata.position.assign(position, position + numberOfParticles);
	tdata.velocity.assign(velocity, velocity + numberOfParticles);
	tdata.positionOut.resize(numberOfParticles);

	// because "nested" thrust-transforms calls are not allowed and any other method would use N² memory I decided to step through the particle array via a CPU for-loop
	for (uint i = 0; i < numberOfParticles; ++i)
	{
		float3 particlePosition = make_float3(position[i]);
		float3 particleVelocity = make_float3(velocity[i]);

		// calculate pressure
		thrust::transform(tdata.position.begin(), tdata.position.end(), tdata.velocity.begin(), tdata.positionOut.begin(), PressureFunctor(particlePosition, particleVelocity, simData));
		//float4 pressure4 = make_float4(0.0);
		float4 pressure4 = thrust::reduce(tdata.positionOut.begin(), tdata.positionOut.end(), make_float4(0.0f), thrust::plus<float4>());//sums all values together

		particleVelocity += (gravity + make_float3(pressure4)) * dt;

		// static collision
		//TODO: write some kind of for-loop
		if ((particlePosition.x + particleVelocity.x * dt > dimension.x && particleVelocity.x > 0.f) || (particlePosition.x + particleVelocity.x * dt < 0.f && particleVelocity.x < 0.f))
		{
			if (particlePosition.x + particleVelocity.x * dt < 0.f)
				particlePosition.x = 0.f;
			else
				particlePosition.x = dimension.x;

			particleVelocity.x *= -.3f;
		}

		if ((particlePosition.y + particleVelocity.y * dt > dimension.y && particleVelocity.y > 0.f) || (particlePosition.y + particleVelocity.y * dt < 0.f && particleVelocity.y < 0.f))
		{
			if (particlePosition.y + particleVelocity.y * dt < 0.f)
				particlePosition.y = 0.f;
			else
				particlePosition.y = dimension.y;

			particleVelocity.y *= -.3f;
		}

		if ((particlePosition.z + particleVelocity.z * dt > dimension.z && particleVelocity.z > 0.f) || (particlePosition.z + particleVelocity.z * dt < 0.f && particleVelocity.z < 0.f))
		{
			if (particlePosition.z + particleVelocity.z * dt < 0.f)
				particlePosition.z = 0.f;
			else
				particlePosition.z = dimension.z;

			particleVelocity.z *= -.3f;
		}
		// *** sc

		positionOut[i] = make_float4(particlePosition + particleVelocity * dt);
		velocity[i] = make_float4(particleVelocity);
	}
	// calculate simulation
	//thrust::transform(devicePos.begin(), devicePos.end(), deviceVel.begin(), deviceOut.begin(), SimulationFunctor(dt, dimension, gravity, simData));
	
	//thrust::copy(tdata.positionOut.begin(), tdata.positionOut.end(), positionOut);
}
