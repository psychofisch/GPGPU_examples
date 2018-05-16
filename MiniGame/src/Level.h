#pragma once

#include <vector>

#include "ParticleSystem.h"
#include "Cube.h"

class Endzone : public Cube
{
public:
	bool isActive();
	bool setActive(bool _b = true);

private:
	bool mIsActive;
};

class Level
{
public:
	Level();
	~Level();

	// build level functions
	void addLevelCollider(const Cube& _c);
	void setReady(bool _r);
	void setParticleSystem(ParticleSystem *_ps);
	void setEndzone(ofVec3f _position, ofVec3f _size);
	void setStartzone(ofVec3f _position, ofVec3f _size, uint _numberOfSpawnParticles);
	void setLevelShader(ofShader* _s);
	void setSunDirection(ofVec3f* _sd);

	// getters
	const std::vector<Cube>& getLevelColliders() const;
	bool isReady();
	bool isRunning();
	bool isPaused();
	uint getScore();
	float getCurrentTime();

	// use level functions
	void resetLevel();
	void draw(ofVec3f& _cameraPos, ofPolyRenderMode _rm);
	void update(float dt);
	void start();
	void end();
	void pause(bool _p);

private:
	std::vector<Cube> mLevelCollider;
	Endzone mEndzone;
	Cube mStartzone;
	ParticleSystem *mExternalParticleSystem;
	ofShader* mLevelShader;
	uint mCurrentScore,
		mNumberOfSpawnParticles;
	float mSecondsToFinishLevel,
		  mCurrentTimeSec;
	ofVec3f* mSunDirection;
	bool mIsReady,
		mDirtyColliders,
		mIsRunning,
		mIsPaused;
	ofTimer mTimer;
};

