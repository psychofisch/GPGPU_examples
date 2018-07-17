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

__host__ __device__ posVel ThrustHelper::SimulationFunctor::operator()(posVel input)
{
	float3 particlePosition = make_float3(input.get<0>());
	float3 particleVelocity = make_float3(input.get<1>());

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

	particlePosition += particleVelocity * dt;

	posVel result;
	result.get<0>() = make_float4(particlePosition);
	result.get<1>() = make_float4(particleVelocity);

	return result;
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
	tdata.position.assign(position, position + numberOfParticles);
	tdata.velocity.assign(velocity, velocity + numberOfParticles);
	tdata.positionOut.resize(numberOfParticles);

	auto inFirst = thrust::make_zip_iterator(thrust::make_tuple(tdata.position.begin(), tdata.velocity.begin()));
	auto inLast = thrust::make_zip_iterator(thrust::make_tuple(tdata.position.end(), tdata.velocity.end()));
	auto outFirst = thrust::make_zip_iterator(thrust::make_tuple(tdata.positionOut.begin(), tdata.velocity.begin()));

	thrust::transform(inFirst, inLast, outFirst, SimulationFunctor(dt, dimension, gravity, simData));

	thrust::copy(tdata.positionOut.begin(), tdata.positionOut.end(), position);
	thrust::copy(tdata.velocity.begin(), tdata.velocity.end(), velocity);
}
