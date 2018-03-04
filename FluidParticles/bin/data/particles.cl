struct SimulationData
{
	float interactionRadius;
	float rho0;//restDensity
	float spring;
	float springNear;
};

float4 calculatePressure(__global float4* position, 
	__global float4* velocity, 
	uint index, 
	float3 pos, 
	float3 vel, 
	uint numberOfParticles,
	struct SimulationData simData);

__kernel void particleUpdate(
	__global float4 *position,
	__global float4 *positionsOut,
	__global float4 *velocity,
	const float dt,
	const float4 gravity,
	const float4 dimension,
	const uint numberOfParticles,
	struct SimulationData simData
)
{
	uint index = get_global_id(0);

	if(index >= numberOfParticles)
		return;
	
	float3 particlePosition = position[index].xyz;
	float3 particleVelocity = velocity[index].xyz;
	float4 particlePressure = calculatePressure(position, velocity, index, particlePosition, particleVelocity, numberOfParticles, simData);
	
	//gravity
	particleVelocity += (gravity.xyz + particlePressure.xyz) * dt;

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
	particlePosition += particleVelocity * dt;

	positionsOut[index] = (float4)(particlePosition, index);
	velocity[index] = (float4)(particleVelocity, numberOfParticles);
	//velocityBuffer[index] = particlePressure;
}

float4 calculatePressure(__global float4* position,
	__global float4* velocity,
	uint index,
	float3 pos,
	float3 vel,
	uint numberOfParticles,
	struct SimulationData simData)
{
	float4 pressureVec = (float4)0.0f;
	float4 viscosityVec = pressureVec;
	for (uint i = 0; i < numberOfParticles; i++)
	{
		if (index == i)
			continue;

		float3 dirVec = pos - position[i].xyz;
		float dist = length(dirVec);//TODO: maybe use half_length

		if (dist > simData.interactionRadius * 1.0f || dist < 0.00001f)
			continue;

		float3 dirVecN = normalize(dirVec);
		float moveDir = dot(vel - velocity[i].xyz, dirVecN);
		float distRel = dist / simData.interactionRadius;

		// viscosity
		if (moveDir > 0)
		{
			float3 impulse = (1.f - distRel) * (simData.spring * moveDir + simData.springNear * moveDir * moveDir) * dirVecN;
			viscosityVec -= (float4)(impulse * 0.5f, 0.f);//goes back to the caller-particle
														//viscosityVec.w = 666.0f;
		}
		// *** v

		float oneminusx = 1.f - distRel;
		float sqx = oneminusx * oneminusx;
		float pressure = 1.f - simData.rho0 * (sqx * oneminusx - sqx);
		//float pressure = amplitude * exp(-dist / interactionRadius);

		pressureVec += (float4)(pressure * dirVecN, 0.f);
		// pressureVec += vec4(dirVec, 0.f);

		pressureVec.w = dist;

		//break;
	}

	return pressureVec + viscosityVec;
}
