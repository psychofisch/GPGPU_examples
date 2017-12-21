#include "ParticleSystem.h"



ParticleSystem::ParticleSystem(uint maxParticles)
	:mGravity(0.f, -9.81f, 0.f),
	mCapacity(maxParticles),
	mNumberOfParticles(0)
{
	mPositions = new ofVec3f[maxParticles];
	mVelocity = new ofVec3f[maxParticles];
	mPressure = new ofVec3f[maxParticles];
}


ParticleSystem::~ParticleSystem()
{
	delete[] mPositions;
	delete[] mVelocity;
	delete[] mPressure;
}

void ParticleSystem::setNumberOfParticles(uint nop)
{
	if (nop > mCapacity || mCapacity == 0)
	{
		return;
	}

	mNumberOfParticles = nop;
}

void ParticleSystem::setDimensions(ofVec3f dimensions)
{
	mDimension = dimensions;
}

void ParticleSystem::setRotation(ofQuaternion rotation)
{
	mRotation = rotation;
}

void ParticleSystem::addRandom(uint particleAmount)
{
	if (mCapacity < mNumberOfParticles + particleAmount)
	{
		std::cout << "no more particles can be spawned!\n";
		return;
	}

	ofSeedRandom();
	for (uint i = 0; i < particleAmount; i++)
	{
		mPositions[mNumberOfParticles + i].x = ofRandom(mDimension.x);
		mPositions[mNumberOfParticles + i].y = ofRandom(mDimension.y);
		mPositions[mNumberOfParticles + i].z = ofRandom(mDimension.z);
	}

	mNumberOfParticles += particleAmount;
}

void ParticleSystem::addDamBreak(uint particleAmount)
{
	ofVec3f damSize = mDimension;
	damSize.x *= 0.2f;
	addCube(ofVec3f(0) + 0.1f, damSize - 0.1f, particleAmount);
}

void ParticleSystem::addCube(ofVec3f cubePos, ofVec3f cubeSize, uint particleAmount)
{
	if (mCapacity < mNumberOfParticles + particleAmount)
	{
		std::cout << "no more particles can be spawned!\n";
		return;
	}

	uint position = 0,
		rows, colums, aisles;
	float x, y, z;
	float gap;
	x = y = z = 0.f;

	aisles = cubeSize.z / powf(particleAmount, 1.f / 3.f);
	rows = cubeSize.x / powf(particleAmount, 1.f / 3.f);
	colums = cubeSize.y / powf(particleAmount, 1.f / 3.f);
	gap = powf(particleAmount, 1.f / 3.f);

	for (uint l = 0; l < aisles; l++)
	{
		for (uint c = 0; c < colums; c++)
		{
			for (uint r = 0; r < rows; r++)
			{
				mPositions[mNumberOfParticles + position].x = cubePos.x + x;
				mPositions[mNumberOfParticles + position].y = cubePos.y + y;
				mPositions[mNumberOfParticles + position].z = cubePos.z + z;

				x += gap;
				position++;

				if (position >= particleAmount)
				{
					mNumberOfParticles += particleAmount;
					return;
				}
			}
			y += gap;
			x = 0.f;
		}
		z += gap;
		y = 0.f;
	}

	mNumberOfParticles += particleAmount;
	std::cout << "too many particles to fit into given space\n";
}

void ParticleSystem::addDrop()
{
	if (mCapacity < mNumberOfParticles + 20)
		return;
	
	mNumberOfParticles += 20;
}

ofVec3f * ParticleSystem::getPositionPtr()
{
	return mPositions;
}

ofVec3f ParticleSystem::getDimensions()
{
	return mDimension;
}

uint ParticleSystem::getNumberOfParticles()
{
	return mNumberOfParticles;
}

uint ParticleSystem::getCapacity()
{
	return mCapacity;
}

void ParticleSystem::update(float dt)
{
	ofVec3f gravityRotated = mGravity;
	if (!mRotation.zeroRotation())
	{
		ofVec3f axis;
		float angle;
		mRotation.getRotate(angle, axis);
		gravityRotated = mGravity.rotate(angle, axis);
	}

	float maxSpeed = 500.f;

	//optimization
	maxSpeed = 1.f / maxSpeed;

//#pragma omp parallel for
	for (int i = 0; i < mNumberOfParticles; ++i)//warning: i can't be uint, because OMP needs an int (fix how?)
	{
		ofVec3f particlePosition = mPositions[i];
		ofVec3f particleVelocity = mVelocity[i];
		ofVec3f particlePressure = i_calculatePressureVector(i);
		float r = 1.f;
		//float r = m_randoms[i];
		//m_rng.seed(i * 815, 1337, 420);

		//std::cout << vectorMath::angleD(vectorMath::normalize(newVel) - vectorMath::normalize(m_velocity[i])) << std::endl;
		//std::cout << vectorMath::radToDeg(atan2f(newVel.y - m_velocity[i].y, newVel.x - m_velocity[i].x)) << std::endl;

		//gravity
		if	  (ofRectangle(-0.1f, -0.1f, mDimension.x + 0.1f, mDimension.y + 0.1f).inside(particlePosition.x, particlePosition.y)
			&& ofRectangle(-0.1f, -0.1f, mDimension.x + 0.1f, mDimension.z + 0.1f).inside(particlePosition.x, particlePosition.z)
			&& ofRectangle(-0.1f, -0.1f, mDimension.y + 0.1f, mDimension.z + 0.1f).inside(particlePosition.y, particlePosition.z))
			particleVelocity += (gravityRotated + particlePressure) * dt;
		//***g

		//static collision
		if ((particlePosition.x > mDimension.x && particleVelocity.x > 0.f) || (particlePosition.x < 0.f && particleVelocity.x < 0.f))
		{
			particleVelocity.x *= -(.1f + 0.2f * r);
		}

		if ((particlePosition.z > mDimension.z && particleVelocity.z > 0.f) || (particlePosition.z < 0.f && particleVelocity.z < 0.f))
			particleVelocity.z *= -(.1f + 0.2f * r);

		if ((particlePosition.y > mDimension.y && particleVelocity.y > 0.f) || (particlePosition.y < 0.f && particleVelocity.y < 0.f))
			particleVelocity.y *= -(.1f + 0.2f * r);
		//*** sc

		//particleVelocity += dt * particleVelocity * -0.01f;//damping
		particlePosition += particleVelocity /** dt*/;

		mPositions[i] = particlePosition;
		mVelocity[i] = particleVelocity;

		//m_vertices[i].position = particlePosition;
	}
}

uint ParticleSystem::debug_testIfParticlesOutside()
{
	uint count = 0;
	for (uint i = 0; i < mNumberOfParticles; ++i)//warning: i can't be uint, because OMP needs an int (fix how?)
	{
		ofVec3f particlePosition = mPositions[i];
		if (particlePosition.x > mDimension.x || particlePosition.x < 0.f
			|| particlePosition.y > mDimension.y || particlePosition.y < 0.f
			|| particlePosition.z > mDimension.z || particlePosition.z < 0.f)
		{
			count++;
			//__debugbreak();
		}
	}
	return count;
}

ofVec3f ParticleSystem::i_calculatePressureVector(size_t index)
{
	float smoothingWidth = pow(30.f, 2);
	float amplitude = 1.f;

	ofVec3f pressureVec;
	for (uint i = 0; i < mNumberOfParticles; ++i)
	{
		if (index == i)
			continue;

		ofVec3f particlePosition = mPositions[i];
		ofVec3f dirVec = mPositions[index] - particlePosition;
		float dist = dirVec.lengthSquared();

		//if (dist > smoothingWidth * .2f)
		//	continue;

		//float pressure = 1.f - (dist/smoothingWidth);
		float pressure = amplitude * expf(-powf(dist, 2) / smoothingWidth);
		//pressureVec += pressure * vectorMath::normalize(dirVec);
		pressureVec += pressure * dirVec;
	}
	return pressureVec;
}
