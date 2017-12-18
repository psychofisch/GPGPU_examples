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
	mGravity = ofVec3f(0.f, -9.81f, 0.f);
	mGravity.rotate(rotation.getEuler().x, rotation.getEuler().y, rotation.getEuler().z);
}

void ParticleSystem::init3DGrid(uint rows, uint colums, uint aisles, float gap)//TODO: dimension
{
	uint position = 0;
	float x, y, z;
	x = y = z = 0.f;
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

				if (position >= mNumberOfParticles * 3)
					return;
			}
			y += gap;
			x = 0.f;
		}
		z += gap;
		y = 0.f;
	}
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
	/*for (uint i = 0; i < mNumberOfParticles; i++)
	{
		if (mPositions[i].y >= 0.f)
			mPositions[i].y -= 3.f * dt;
	}*/

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

		//gravity
		/*if (particlePosition.y > 0)
			particleVelocity.y -= 9.81f * dt;*/
		/*if    (((particlePosition.x < mDimension.x) && (particlePosition.x > 0.f))
			&& ((particlePosition.y < mDimension.y) && (particlePosition.y > 0.f))
			&& ((particlePosition.z < mDimension.z) && (particlePosition.z > 0.f)))
			particleVelocity += mGravity * dt;*/

		if (ofRectangle(0, 0, mDimension.x, mDimension.y).inside(particlePosition.x, particlePosition.y)
			&& ofRectangle(0, 0, mDimension.x, mDimension.z).inside(particlePosition.x, particlePosition.z)
			&& ofRectangle(0, 0, mDimension.y, mDimension.z).inside(particlePosition.y, particlePosition.z))
			particleVelocity += (mGravity + particlePressure) * dt;
		//***g

		particleVelocity -= dt * particleVelocity * 0.25f;//damping
		particlePosition += particleVelocity /** dt*/;

		mPositions[i] = particlePosition;
		mVelocity[i] = particleVelocity;

		//m_vertices[i].position = particlePosition;
	}
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
