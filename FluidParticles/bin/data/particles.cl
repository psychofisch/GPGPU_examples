float4 calculatePressure(__global float4 *positions,
	uint index,
	uint numberOfParticles,
	float interactionRadius)
{
	float3 particlePosition = positions[index].xyz;
	
	float4 pressureVec = (float4)0.0f;
	for (uint i = 0; i < numberOfParticles; i++)
	{
		if (index == i)
			continue;

		float3 dirVec = particlePosition - positions[i].xyz;
		float dist = length(dirVec);//TODO: maybe use half_length

		// if(particlePosition == positionBuffer[i].xyz)
		// {
			// pressureVec = vec4(positionBuffer[i].xyz, i);
			// break;
		// }
		//if (dist > interactionRadius * 1.0f || dist < 0.01f)
		if (dist > interactionRadius * 1.0f)
			continue;

		float pressure = 1.f - (dist/interactionRadius);
		//float pressure = amplitude * exp(-dist / interactionRadius);
		
		pressureVec += (float4)(pressure * normalize(dirVec), 0.f);
		// pressureVec += vec4(dirVec, 0.f);
		
		pressureVec.w = dist;
		
		//break;
	}
	
	return pressureVec;
}

__kernel void particleUpdate(
	__global float4 *positions,
	__global float4 *positionsOut,
	__global float4 *velocity,
	const float dt,
	const float interactionRadius,
	const float4 gravity,
	const float4 dimension,
	const uint numberOfParticles
)
{
	uint index = get_global_id(0);

	if(index >= numberOfParticles)
		return;
	
	float3 particlePosition = positions[index].xyz;
	float3 particleVelocity = velocity[index].xyz;
	float4 particlePressure = calculatePressure(positions, index, numberOfParticles, interactionRadius);
	
	if (   particlePosition.x <= dimension.x || particlePosition.x >= 0.f
		|| particlePosition.y <= dimension.y || particlePosition.y >= 0.f
		|| particlePosition.z <= dimension.z || particlePosition.z >= 0.f)
		particleVelocity += (gravity.xyz + particlePressure.xyz) * dt;

	// static collision
	//TODO: write some kind of for-loop
	if ((particlePosition.x + particleVelocity.x * dt > dimension.x && particleVelocity.x > 0.f) || (particlePosition.x + particleVelocity.x * dt < 0.f && particleVelocity.x < 0.f))
	{
		particleVelocity.x *= -.3f;
	}
	
	if ((particlePosition.y + particleVelocity.y * dt > dimension.y && particleVelocity.y > 0.f) || (particlePosition.y + particleVelocity.y * dt < 0.f && particleVelocity.y < 0.f))
	{
		particleVelocity.y *= -.3f;
	}
	
	if ((particlePosition.z + particleVelocity.z * dt > dimension.z && particleVelocity.z > 0.f) || (particlePosition.z + particleVelocity.z * dt < 0.f && particleVelocity.z < 0.f))
	{
		particleVelocity.z *= -.3f;
	}
	// *** sc

	// particleVelocity += dt * particleVelocity * -0.01f;//damping
	particlePosition += particleVelocity * dt;

	positionsOut[index] = (float4)(particlePosition, index);
	velocity[index] = (float4)(particleVelocity, numberOfParticles);
	//velocityBuffer[index] = particlePressure;
}
