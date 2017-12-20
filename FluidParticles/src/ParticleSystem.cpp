#include "ParticleSystem.h"



ParticleSystem::ParticleSystem()
	:mGravity(0.f, -9.81f, 0.f)
{
}


ParticleSystem::~ParticleSystem()
{
	delete[] mPositions;
	delete[] mVelocity;
	delete[] mPressure;
}

void ParticleSystem::setDimensions(ofVec3f dimensions)
{
	mDimension = dimensions;
}

void ParticleSystem::setNumberOfParticles(uint nop)
{
	mNumberOfParticles = nop;
	mPositions = new ofVec3f[mNumberOfParticles];
	mVelocity = new ofVec3f[mNumberOfParticles];
	mPressure = new ofVec3f[mNumberOfParticles];
}

void ParticleSystem::setRotation(ofQuaternion rotation)
{
	mRotation = rotation;
}

void ParticleSystem::init3DGrid()//TODO: dimension
{
	uint position = 0,
		rows, colums, aisles;
	float x, y, z;
	float gap;
	x = y = z = 0.01f;

	aisles = mDimension.z / powf(mNumberOfParticles, 1.f / 3.f);
	rows = mDimension.x / powf(mNumberOfParticles, 1.f / 3.f);
	colums = mDimension.y / powf(mNumberOfParticles, 1.f / 3.f);
	gap = powf(mNumberOfParticles, 1.f / 3.f) * 0.95f;

	for (uint l = 0; l < aisles; l++)
	{
		for (uint c = 0; c < colums; c++)
		{
			for (uint r = 0; r < rows; r++)
			{
				mPositions[position].x = x;
				mPositions[position].y = y;
				mPositions[position].z = z;

				x += gap;
				position++;

				if (x > mDimension.x || y > mDimension.y || z > mDimension.z)
					__debugbreak();

				if (position >= mNumberOfParticles)
					return;
			}
			y += gap;
			x = 0.01f;
		}
		z += gap;
		y = 0.01f;
	}

	std::cout << "too many particles to fit into given space\n";
}

void ParticleSystem::initRandom()
{
	ofSeedRandom();
	for (uint i = 0; i < mNumberOfParticles; i++)
	{
		mPositions[i].x = ofRandom(mDimension.x);
		mPositions[i].y = ofRandom(mDimension.y);
		mPositions[i].z = ofRandom(mDimension.z);
	}
}

void ParticleSystem::initDamBreak()
{
	float tmp = mDimension.x;
	mDimension.x *= 0.2f;
	init3DGrid();
	mDimension.x = tmp;
}

void ParticleSystem::addDrop()
{
	
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
		if (ofRectangle(0, 0, mDimension.x, mDimension.y).inside(particlePosition.x, particlePosition.y)
			&& ofRectangle(0, 0, mDimension.x, mDimension.z).inside(particlePosition.x, particlePosition.z)
			&& ofRectangle(0, 0, mDimension.y, mDimension.z).inside(particlePosition.y, particlePosition.z))
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
