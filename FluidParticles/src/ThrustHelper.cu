#include "ThrustHelper.h"

inline __host__ __device__ bool operator==(float3& lhs, float3& rhs)
{
	if (lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z)
		return true;
	else
		return false;
}

ThrustHelper::SimulationFunctor::SimulationFunctor(float dt_, float3 dim_, float3 g_, SimulationData simData_, const MinMaxDataCuda* colliders_, uint numberOfColliders_)
	:dt(dt_),
	dimension(dim_),
	gravity(g_),
	simData(simData_),
	colliders(colliders_),
	numberOfColliders(numberOfColliders_)
{}

__host__ __device__ posVel ThrustHelper::SimulationFunctor::operator()(posVel input)
{
	float3 particlePosition = make_float3(input.get<0>());
	float3 particleVelocity = make_float3(input.get<1>());

	float fluidDamp = 0.0;
	float particleSize = simData.interactionRadius * 0.1f;
	float3 worldAABBmin = make_float3(particleSize);
	float3 worldAABBmax = dimension - particleSize;

	//float3 particlePressure = calculatePressure(positions, velocity, index, particlePosition, particleVelocity, numberOfParticles, simData);
	float3 particlePressure = make_float3(0.f);

	// gravity
	particleVelocity += (gravity + particlePressure) * dt;
	// *** g

	float3 deltaVelocity = particleVelocity * dt;
	float3 sizeOffset = normalize(particleVelocity) * particleSize;
	float3 newPos = particlePosition + deltaVelocity;

	// static collision
	int collisionCnt = 3; //support multiple collisions
	for (int i = 0; i < numberOfColliders && collisionCnt > 0; i++)
	{
		MinMaxDataCuda currentAABB = colliders[i];
		float3 intersection = make_float3(0.f);
		float fraction;
		bool result = false;

		// ERROR: LineAABBIntersection from the CUDA implementation can't be used
		result = LineAABBIntersection(currentAABB, particlePosition, newPos + sizeOffset, intersection, fraction);

		if (result == false)
			continue;

		if (intersection.x == currentAABB.max.x || intersection.x == currentAABB.min.x)
			particleVelocity.x *= -fluidDamp;
		else if (intersection.y == currentAABB.max.y || intersection.y == currentAABB.min.y)
			particleVelocity.y *= -fluidDamp;
		else if (intersection.z == currentAABB.max.z || intersection.z == currentAABB.min.z)
			particleVelocity.z *= -fluidDamp;
		//else
		//	std::cout << "W00T!?\n";//DEBUG

		//particlePosition = intersection;
		newPos = intersection;
		break;// DEBUG! this prevents multiple collisions!
	}
	// *** sc

	// ERROR: the "dim" function from the CUDA implementation can't be used
	// bounding box collision
	float3 tmpVel = particleVelocity;
	for (int i = 0; i < 3; ++i)
	{
		if ((dim(newPos, i) > dim(worldAABBmax, i) && dim(tmpVel, i) > 0.0) // max boundary
			|| (dim(newPos, i) < dim(worldAABBmin, i) && dim(tmpVel, i) < 0.0) // min boundary
			)
		{
			/*if (newPos[i] < worldAABB.min[i])
			newPos[i] = worldAABB.min[i];
			else
			newPos[i] = worldAABB.max[i];*/

			dim(tmpVel, i) *= -fluidDamp;
		}
	}

	particleVelocity = tmpVel;
	// *** bbc

	// particleVelocity += dt * particleVelocity * -0.01f;//damping
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
	const MinMaxDataCuda* staticColliders,
	const float dt,
	const float3 gravity,
	const float3 dimension,
	const uint numberOfParticles,
	const uint numberOfColliders,
	SimulationData simData)
{
	tdata.position.assign(position, position + numberOfParticles);
	tdata.velocity.assign(velocity, velocity + numberOfParticles);
	tdata.positionOut.resize(numberOfParticles);

	auto inFirst = thrust::make_zip_iterator(thrust::make_tuple(tdata.position.begin(), tdata.velocity.begin()));
	auto inLast = thrust::make_zip_iterator(thrust::make_tuple(tdata.position.end(), tdata.velocity.end()));
	auto outFirst = thrust::make_zip_iterator(thrust::make_tuple(tdata.positionOut.begin(), tdata.velocity.begin()));

	thrust::transform(inFirst, inLast, outFirst, SimulationFunctor(dt, dimension, gravity, simData, staticColliders, numberOfColliders));

	thrust::copy(tdata.positionOut.begin(), tdata.positionOut.end(), position);
	thrust::copy(tdata.velocity.begin(), tdata.velocity.end(), velocity);
}

//void ThrustHelper::thrustCopyColliders(ThrustParticleData & tdata, std::vector<MinMaxDataCuda>& collision)
//{
//}
