#pragma once

#include "ofVec3f.h"
#include "ofRectangle.h"
#include "ofMath.h"
#include "ofQuaternion.h"

typedef unsigned int uint;

class ParticleSystem
{
public:
	ParticleSystem();
	~ParticleSystem();

	void setDimensions(ofVec3f dimensions);
	void setNumberOfParticles(uint nop);
	void setRotation(ofQuaternion rotation);
	void init3DGrid();
	void initRandom();
	void initDamBreak();
	ofVec3f* getPositionPtr();
	ofVec3f getDimensions();
	uint getNumberOfParticles();
	void update(float dt);

private:
	uint mNumberOfParticles;
	ofVec3f	*mPositions,
		*mVelocity,
		*mPressure,
		mDimension,
		mGravity;

	ofVec3f i_calculatePressureVector(size_t index);
};

