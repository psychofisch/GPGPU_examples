#include "ThrustHelper.h"

inline __host__ __device__ bool operator==(float3& lhs, float3& rhs)
{
	if (lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z)
		return true;
	else
		return false;
}

ThrustHelper::PressureFunctor::PressureFunctor(float3 pos_, float3 vel_, SimulationData simData_)
	:pos(pos_),
	vel(vel_),
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

ThrustHelper::CollisionFunctor::CollisionFunctor(uint aoc_, MinMaxDataThrust* mmd_)
	:amountOfCubes(aoc_),
	minMaxRaw(mmd_)
{}

__host__ __device__ int ThrustHelper::CollisionFunctor::operator()(MinMaxDataThrust minMax)
{
	int result = -1;
	float3 currentMin = minMax.min;
	float3 currentMax = minMax.max;
	int i = minMax.id;
	for (uint j = 0; j < amountOfCubes; j++)
	{
		if (i == j)
			continue;

		float3 otherMin = minMaxRaw[j].min;
		float3 otherMax = minMaxRaw[j].max;

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


ThrustHelper::ThrustParticleData* ThrustHelper::setup(uint numberOfParticles)
{
	ThrustParticleData* r = new ThrustParticleData;
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
		thrust::plus<float4> plus;
		float4 pressure4 = thrust::reduce(tdata.positionOut.begin(), tdata.positionOut.end(), make_float4(0.f), plus);

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

		// particleVelocity += dt * particleVelocity * -0.01f;//damping
		//tdata.positionOut[i] = make_float4(particlePosition + particleVelocity * dt);
		positionOut[i] = make_float4(particlePosition + particleVelocity * dt);
		velocity[i] = make_float4(particleVelocity);
	}
	// calculate simulation
	//thrust::transform(devicePos.begin(), devicePos.end(), deviceVel.begin(), deviceOut.begin(), SimulationFunctor(dt, dimension, gravity, simData));
	
	//thrust::copy(tdata.positionOut.begin(), tdata.positionOut.end(), positionOut);
}

void ThrustHelper::thrustGetCollisions(
	ThrustCollisionData& tdata,
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
	thrust::transform(
		tdata.minMaxBuffer.begin(),
		tdata.minMaxBuffer.end(),
		tdata.collisions.begin(),
		CollisionFunctor(amountOfCubes, minMaxDevice)
	);

	// copy the device collision buffer to the host
	thrust::copy(tdata.collisions.begin(), tdata.collisions.end(), collisionBuffer);
}
