struct SimulationData
{
	float interactionRadius;
	float pressureMultiplier;
	float viscosity;
	float restPressure;
};

struct MinMaxData
{
	float4 min, max;
};

float dim4(const float4* v, size_t i)
{
	switch (i)
	{
	case 0: return (*v).x;//no breaks required
	case 1:	return (*v).y;
	case 2:	return (*v).z;
	case 3: return (*v).w;
	}

	return NAN;
}

float dim3(const float3* v, size_t i)
{
	switch (i)
	{
	case 0: return (*v).x;//no breaks required
	case 1:	return (*v).y;
	case 2:	return (*v).z;
	}			   

	return NAN;
}

//inline float* dim3ptr(float3* v, size_t i)
//{
//	switch (i)
//	{
//	case 0: return &((*v).x);//no breaks required
//	case 1:	return &((*v).y);
//	case 2:	return &((*v).z);
//	}
//
//	return v.x;//TODO: return something that indicates an error
//}

float3 calculatePressure(__global float4* position, 
	__global float4* velocity, 
	uint index, 
	float3 pos, 
	float3 vel, 
	uint numberOfParticles,
	struct SimulationData simData);

bool ClipLine(int d, const struct MinMaxData aabbBox, const float3 v0, const float3 v1, float* f_low, float* f_high);
bool LineAABBIntersection(const struct MinMaxData aabbBox, const float3 v0, const float3 v1, float3* vecIntersection, float* flFraction);

__kernel void particleUpdate(
	__global float4 *positions,
	__global float4 *positionsOut,
	__global float4 *velocity,
	__global struct MinMaxData* staticColliders,
	const float dt,
	const float4 gravity,
	const float4 position,
	const float4 dimension,
	const uint numberOfParticles,
	const uint numberOfColliders,
	struct SimulationData simData
)
{
	uint index = get_global_id(0);

	if(index >= numberOfParticles)
		return;

	float fluidDamp = 0.0;
	float particleSize = simData.interactionRadius * 0.1f;
	float3 worldAABBmin = particleSize;
	float3 worldAABBmax = dimension.xyz - particleSize;

	float3 particlePosition = positions[index].xyz;
	float3 particleVelocity = velocity[index].xyz;
	float3 particlePressure = calculatePressure(positions, velocity, index, particlePosition, particleVelocity, numberOfParticles, simData);
	
	particlePosition -= position.xyz;

	//gravity
	particleVelocity += (gravity.xyz + particlePressure.xyz) * dt;

	float3 deltaVelocity = particleVelocity * dt;
	float3 sizeOffset = normalize(particleVelocity) * particleSize;
	float3 newPos = particlePosition + deltaVelocity;

	// static collision
	int collisionCnt = 3; //support multiple collisions
	for (int i = 0; i < numberOfColliders && collisionCnt > 0; i++)
	{
		struct MinMaxData currentAABB = staticColliders[i];
		float3 intersection;
		float fraction;
		bool result = false;


		result = LineAABBIntersection(currentAABB, particlePosition, newPos + sizeOffset, &intersection, &fraction);

		if (result == false)
			continue;

		if (intersection.x == currentAABB.max.x || intersection.x == currentAABB.min.x)
			particleVelocity.x *= -1.0;
		else if (intersection.y == currentAABB.max.y || intersection.y == currentAABB.min.y)
			particleVelocity.y *= -1.0;
		else if (intersection.z == currentAABB.max.z || intersection.z == currentAABB.min.z)
			particleVelocity.z *= -1.0;
		//else
		//	std::cout << "W00T!?\n";//DEBUG

		//particlePosition = intersection;
		newPos = intersection;
		break;// DEBUG! this prevents multiple collisions!

			  //	//ofVec3f reflection;
			  //	ofVec3f n = Particle::directions[closest];

			  //	// source -> https://math.stackexchange.com/questions/13261/how-to-get-a-reflection-vector#13266
			  //	particleVelocity = particleVelocity - (2 * particleVelocity.dot(n) * n);
			  //	particleVelocity *= fluidDamp;
			  //	
			  //	collisionCnt = 0;
			  //	//result = j;
			  //	//break;// OPT: do not delete this (30% performance loss)
	}
	// *** sc

	// bounding box collision
	//TODO: write some kind of for-loop
	if ((newPos.x > worldAABBmax.x && particleVelocity.x > 0.f) || (newPos.x < worldAABBmin.x && particleVelocity.x < 0.f))
	{
		particleVelocity.x *= -fluidDamp;
	}
	
	if ((newPos.y > worldAABBmax.y && particleVelocity.y > 0.f) || (newPos.y < worldAABBmin.x && particleVelocity.y < 0.f))
	{
		particleVelocity.y *= -fluidDamp;
	}
	
	if ((newPos.z > worldAABBmax.z && particleVelocity.z > 0.f) || (newPos.z < worldAABBmin.z && particleVelocity.z < 0.f))
	{
		particleVelocity.z *= -fluidDamp;
	}
	// *** sc

	// particleVelocity += dt * particleVelocity * -0.01f;//damping
	particlePosition += particleVelocity * dt;

	positionsOut[index] = (float4)(particlePosition + position.xyz, 0.f);
	velocity[index] = (float4)(particleVelocity, numberOfParticles);
	//velocityBuffer[index] = particlePressure;
}

float3 calculatePressure(__global float4* position,
	__global float4* velocity,
	uint index,
	float3 pos,
	float3 vel,
	uint numberOfParticles,
	struct SimulationData simData)
{
	float interactionRadius = simData.interactionRadius;
	float3 particlePosition = pos;
	//float density = 0.f;
	float viscosity = simData.viscosity;
	float restPressure = simData.restPressure;
	float pressureMultiplier = simData.pressureMultiplier;
	float influence = 0.f;

	float3 pressureVec = (float3)0.0f;
	float3 viscosityVec = pressureVec;

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
		float distRel = 1.0f - (dist / simData.interactionRadius);

		float sqx = distRel * distRel;

		influence += 1.0f;

		// viscosity
		if (true || moveDir > 0)
		{
			//ofVec3f impulse = (1.f - distRel) * (mSimData.viscosity * moveDir + mSimData.restPressure * moveDir * moveDir) * dirVecN;
			float factor = sqx * (viscosity * moveDir);
			float3 impulse = factor * dirVecN;
			viscosityVec -= impulse;
		}
		// *** v

		// pressure 
		float pressure = sqx * pressureMultiplier;
		// *** p

		//density += 1.f;

		pressureVec += (pressure - restPressure) * dirVecN;
	}

	if (influence > 0.f)
	{
		viscosityVec = viscosityVec / influence;
	}

	if (length(viscosityVec) > 100.0f)
		viscosityVec = normalize(viscosityVec) * 100.0f;

	return pressureVec + viscosityVec;
}

bool ClipLine(int d, const struct MinMaxData aabbBox, const float3 v0, const float3 v1, float* f_low, float* f_high)
{
	// f_low and f_high are the results from all clipping so far. We'll write our results back out to those parameters.

	// f_dim_low and f_dim_high are the results we're calculating for this current dimension.
	float f_dim_low, f_dim_high;

	// Find the point of intersection in this dimension only as a fraction of the total vector http://youtu.be/USjbg5QXk3g?t=3m12s
	f_dim_low = (dim4(&aabbBox.min, d) - dim3(&v0, d)) / (dim3(&v1, d) - dim3(&v0, d));
	f_dim_high = (dim4(&aabbBox.max, d) - dim3(&v0, d)) / (dim3(&v1, d) - dim3(&v0, d));

	// Make sure low is less than high
	if (f_dim_high < f_dim_low)
	{
		float tmp = f_dim_high;
		f_dim_high = f_dim_low;
		f_dim_low = tmp;
	}

	// If this dimension's high is less than the low we got then we definitely missed. http://youtu.be/USjbg5QXk3g?t=7m16s
	if (f_dim_high < *f_low)
		return false;

	// Likewise if the low is less than the high.
	if (f_dim_low > *f_high)
		return false;

	// Add the clip from this dimension to the previous results http://youtu.be/USjbg5QXk3g?t=5m32s
	*f_low = max(f_dim_low, *f_low);
	*f_high = min(f_dim_high, *f_high);

	if (*f_low > *f_high)
		return false;

	return true;
}

// Find the intersection of a line from v0 to v1 and an axis-aligned bounding box http://www.youtube.com/watch?v=USjbg5QXk3g
bool LineAABBIntersection(const struct MinMaxData aabbBox, const float3 v0, const float3 v1, float3* vecIntersection, float* flFraction)
{
	float f_low = 0;
	float f_high = 1;

	if (!ClipLine(0, aabbBox, v0, v1, &f_low, &f_high))
		return false;

	if (!ClipLine(1, aabbBox, v0, v1, &f_low, &f_high))
		return false;

	if (!ClipLine(2, aabbBox, v0, v1, &f_low, &f_high))
		return false;

	// The formula for I: http://youtu.be/USjbg5QXk3g?t=6m24s
	float3 b = v1 - v0;
	*vecIntersection = v0 + b * f_low;

	*flFraction = f_low;

	return true;
}
