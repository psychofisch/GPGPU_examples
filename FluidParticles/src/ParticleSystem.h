#pragma once

// Standardized MAX, MIN and CLAMP
#define MAX(a, b) ((a > b) ? a : b)
#define MIN(a, b) ((a < b) ? a : b)
#define CLAMP(a, b, c) MIN(MAX(a, b), c)    // double sided clip of input a
#define TOPCLAMP(a, b) (a < b ? a:b)	    // single top side clip of input a

#include <oclUtils.h>

#include "ofVec3f.h"
#include "ofRectangle.h"
#include "ofMath.h"
#include "ofQuaternion.h"
#include "ofShader.h"
#include "ofVbo.h"

typedef unsigned int uint;

struct SimulationData
{
	float smoothingWidth;
};

struct ComputeShaderData
{
	ofShader computeShader;
	ofBufferObject positionBuffer;
	ofBufferObject positionOutBuffer;
	ofBufferObject velocityBuffer;
};

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
	void setSmoothingWidth(float sw);
	void addDamBreak(uint particleAmount);
	void addRandom(uint particleAmount);
	void addCube(ofVec3f position, ofVec3f size, uint particleAmount);
	void addDrop();
	void draw();
	ofVec4f* getPositionPtr();
	ofVec3f getDimensions();
	uint getNumberOfParticles();
	uint getCapacity();
	ComputeModes getMode();
	void update(float dt);
	uint debug_testIfParticlesOutside();

private:
	uint mNumberOfParticles,
		mCapacity;
	ComputeModes mMode;
	ofVec4f	*mPositions,
		*mVelocity;
		//*mPressure,
	ofVec3f	mDimension,
		mGravity,
		mGravityRotated;
	ofVbo mParticlesVBO;
	ofQuaternion mRotation;
	ComputeShaderData mComputeData;
	SimulationData mSimData;

	void iUpdateCPU(float dt);
	void iUpdateCompute(float dt);
	ofVec3f iCalculatePressureVector(size_t index);
	//bool mShaderStorageSwap;
};

