#pragma once

#include "ofVec3f.h"
#include "ofRectangle.h"
#include "ofMath.h"

typedef unsigned int uint;

class ParticleSystem
{
public:
	ParticleSystem();
	~ParticleSystem();

	void setDimensions(ofVec3f dimensions);
	void setNumberOfParticles(unsigned int nop);
	void init3DGrid(uint rows, uint colums, uint aisles, float gap);
	void initRandom();
	ofVec3f* getPositionPtr();
	void update(float dt);

private:
	uint mNumberOfParticles;
	ofVec3f	*mPositions,
		*mVelocity,
		*mPressure,
		mDimension;

	ofVec3f i_calculatePressureVector(size_t index);
};

