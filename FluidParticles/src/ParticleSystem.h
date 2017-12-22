#pragma once

#include "ofVec3f.h"
#include "ofRectangle.h"
#include "ofMath.h"
#include "ofQuaternion.h"
#include "ofShader.h"
#include "ofVbo.h"

typedef unsigned int uint;

class ParticleSystem
{
public:
	enum class ComputeModes
	{
		CPU,
		COMPUTE_SHADER,
		CUDA,
		OPENCL
	};

	ParticleSystem(uint mp);
	~ParticleSystem();

	void setDimensions(ofVec3f dimensions);
	void setNumberOfParticles(uint nop);
	void setRotation(ofQuaternion rotation);
	void setMode(ComputeModes m);
	void addDamBreak(uint particleAmount);
	void addRandom(uint particleAmount);
	void addCube(ofVec3f position, ofVec3f size, uint particleAmount);
	void addDrop();
	void draw();
	ofVec3f* getPositionPtr();
	ofVec3f getDimensions();
	uint getNumberOfParticles();
	uint getCapacity();
	void update(float dt);
	uint debug_testIfParticlesOutside();

private:
	uint mNumberOfParticles,
		mCapacity;
	ComputeModes mMode;
	ofVec3f	*mPositions,
		*mVelocity,
		*mPressure,
		mDimension,
		mGravity,
		mGravityRotated;
	ofVbo mParticlesVBO;
	ofQuaternion mRotation;
	ofShader computeShader;
	ofBufferObject mPositionBuffer;

	void iUpdateCPU(float dt);
	ofVec3f iCalculatePressureVector(size_t index);
};

