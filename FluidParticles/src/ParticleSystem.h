#pragma once

#include "ofVec3f.h"
#include "ofRectangle.h"
#include "ofMath.h"
#include "ofQuaternion.h"

typedef unsigned int uint;

class ParticleSystem
{
public:
	ParticleSystem(uint mp);
	~ParticleSystem();

	void setDimensions(ofVec3f dimensions);
	void setNumberOfParticles(uint nop);
	void setRotation(ofQuaternion rotation);
	void addDamBreak(uint particleAmount);
	void addRandom(uint particleAmount);
	void addCube(ofVec3f position, ofVec3f size, uint particleAmount);
	void addDrop();
	ofVec3f* getPositionPtr();
	ofVec3f getDimensions();
	uint getNumberOfParticles();
	uint getCapacity();
	void update(float dt);
	uint debug_testIfParticlesOutside();

private:
	uint mNumberOfParticles,
		mCapacity;
	ofVec3f	*mPositions,
		*mVelocity,
		*mPressure,
		mDimension,
		mGravity;
	ofQuaternion mRotation;

	ofVec3f i_calculatePressureVector(size_t index);
};

