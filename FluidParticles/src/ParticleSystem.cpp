#include "ParticleSystem.h"



ParticleSystem::ParticleSystem(uint maxParticles)
	:mGravity(0.f, -9.81f, 0.f),
	mCapacity(maxParticles),
	mNumberOfParticles(0),
	mMode(ComputeModes::CPU)
{
	mPositions = new ofVec3f[maxParticles];
	mVelocity = new ofVec3f[maxParticles];
	mPressure = new ofVec3f[maxParticles];

	mPositionOutBuffer.allocate(sizeof(ofVec3f) * maxParticles, GL_DYNAMIC_DRAW);
	mPositionBuffer.allocate(sizeof(ofVec3f) * maxParticles, GL_DYNAMIC_DRAW);
	mParticlesVBO.setVertexBuffer(mPositionBuffer, 3, sizeof(ofVec3f), 0);

	if (!mComputeShader.setupShaderFromFile(GL_COMPUTE_SHADER, "particles.compute"))
		exit(1);
	mComputeShader.linkProgram();
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

void ParticleSystem::setMode(ComputeModes m)
{
	mMode = m;
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

	float volume = cubeSize.x * cubeSize.y * cubeSize.z;
	float spacePerParticle = volume / particleAmount;
	float cubedParticles = powf(particleAmount, 1.f / 3.f) * 3;
	float ratio;
	ratio = cubeSize.x / (cubeSize.x + cubeSize.y + cubeSize.z);
	float gap;
	gap = cubeSize.x / (cubedParticles * ratio);

	ofVec3f partPos(0.f);

	for (int i = 0; i < particleAmount; i++)
	{
		mPositions[mNumberOfParticles + i] = cubePos + partPos;
		mVelocity[mNumberOfParticles + i] = ofVec3f(0.f);

		partPos.x += gap;

		if (partPos.x > cubeSize.x)
		{
			partPos.x = 0.f;
			partPos.z += gap;

			if (partPos.z > cubeSize.z)
			{
				partPos.z = 0.f;
				partPos.y += gap;

				if (partPos.y > cubeSize.y)
				{
					std::cout << "addCube: w00t? only " << i << " particles?\n";
					break;
				}
			}
		}
	}

	mNumberOfParticles += particleAmount;
}

void ParticleSystem::addDrop()
{
	if (mCapacity < mNumberOfParticles + 20)
		return;
	
	mNumberOfParticles += 20;
}

void ParticleSystem::draw()
{
	mParticlesVBO.draw(GL_POINTS, 0, mNumberOfParticles);
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
	mGravityRotated = mGravity;
	if (!mRotation.zeroRotation())
	{
		ofVec3f axis;
		float angle;
		mRotation.getRotate(angle, axis);
		mGravityRotated = mGravity.rotate(angle, axis);
	}

	switch (mMode)
	{
	case ComputeModes::CPU:
		iUpdateCPU(dt);
		break;
	case ComputeModes::COMPUTE_SHADER:
		iUpdateCompute(dt);
		break;
	}
}

void ParticleSystem::iUpdateCPU(float dt)
{
	float maxSpeed = 500.f;

	//optimization
	maxSpeed = 1.f / maxSpeed;

#pragma omp parallel for
	for (int i = 0; i < mNumberOfParticles; ++i)//warning: i can't be uint, because OMP needs an int (fix how?)
	{
		ofVec3f particlePosition = mPositions[i];
		ofVec3f particleVelocity = mVelocity[i];
		ofVec3f particlePressure = iCalculatePressureVector(i);
		float r = 1.f;
		//float r = m_randoms[i];
		//m_rng.seed(i * 815, 1337, 420);

		//std::cout << vectorMath::angleD(vectorMath::normalize(newVel) - vectorMath::normalize(m_velocity[i])) << std::endl;
		//std::cout << vectorMath::radToDeg(atan2f(newVel.y - m_velocity[i].y, newVel.x - m_velocity[i].x)) << std::endl;

		//gravity
		/*if	  (ofRectangle(-0.1f, -0.1f, mDimension.x + 0.1f, mDimension.y + 0.1f).inside(particlePosition.x, particlePosition.y)
		&& ofRectangle(-0.1f, -0.1f, mDimension.x + 0.1f, mDimension.z + 0.1f).inside(particlePosition.x, particlePosition.z)
		&& ofRectangle(-0.1f, -0.1f, mDimension.y + 0.1f, mDimension.z + 0.1f).inside(particlePosition.y, particlePosition.z))
		*/
		if (particlePosition.x <= mDimension.x || particlePosition.x >= 0.f
			|| particlePosition.y <= mDimension.y || particlePosition.y >= 0.f
			|| particlePosition.z <= mDimension.z || particlePosition.z >= 0.f)
			particleVelocity += (mGravityRotated + particlePressure) * dt;
		//***g

		//static collision
		if ((particlePosition.x > mDimension.x && particleVelocity.x > 0.f) || (particlePosition.x < 0.f && particleVelocity.x < 0.f))
		{
			particleVelocity.x *= -(.1f + 0.2f * r);
		}

		if ((particlePosition.z > mDimension.z && particleVelocity.z > 0.f) || (particlePosition.z < 0.f && particleVelocity.z < 0.f))
		{
			particleVelocity.z *= -(.1f + 0.2f * r);
		}

		if ((particlePosition.y > mDimension.y && particleVelocity.y > 0.f) || (particlePosition.y < 0.f && particleVelocity.y < 0.f))
		{
			particleVelocity.y *= -(.1f + 0.2f * r);
		}
		//*** sc

		//particleVelocity += dt * particleVelocity * -0.01f;//damping
		particlePosition += particleVelocity * dt * 15.f;//timeScaling

		mVelocity[i] = particleVelocity;
		mPositions[i] = particlePosition;

		//m_vertices[i].position = particlePosition;
	}

	mPositionBuffer.updateData(mNumberOfParticles * sizeof(ofVec3f), mPositions);
}

void ParticleSystem::iUpdateCompute(float dt)
{
	//mPositionBuffer.updateData(sizeof(ofVec3f) * mNumberOfParticles, mPositions);

	mPositionBuffer.bindBase(GL_SHADER_STORAGE_BUFFER, 0);
	mPositionOutBuffer.bindBase(GL_SHADER_STORAGE_BUFFER, 1);

	mComputeShader.begin();
	mComputeShader.dispatchCompute(mNumberOfParticles, 1, 1);
	mComputeShader.end();

	mPositionOutBuffer.copyTo(mPositionBuffer);

	ofVec3f* tmpPositionFromGPU = mPositionBuffer.map<ofVec3f>(GL_READ_ONLY);
	mPositionBuffer.unmap();
}

ofVec3f ParticleSystem::iCalculatePressureVector(size_t index)
{
	float smoothingWidth = pow(20.f, 2);
	//float smoothingWidth = 18.f;
	float amplitude = 1.f;

	ofVec3f pressureVec;
	for (uint i = 0; i < mNumberOfParticles; ++i)
	{
		if (index == i)
			continue;

		ofVec3f particlePosition = mPositions[i];
		ofVec3f dirVec = mPositions[index] - particlePosition;
		float dist = dirVec.lengthSquared();

		if (dist > smoothingWidth * 1.f)
			continue;

		//float pressure = 1.f - (dist/smoothingWidth);
		float pressure = amplitude * expf(-powf(dist, 2) / smoothingWidth);
		//pressureVec += pressure * vectorMath::normalize(dirVec);
		pressureVec += pressure * dirVec;
	}
	return pressureVec;
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
