#include "Level.h"

Level::Level()
	:mExternalParticleSystem(nullptr),
	mCurrentScore(0),
	mIsReady(false),
	mIsRunning(false)
{
}

Level::~Level()
{
}

void Level::addLevelCollider(const Cube & _c)
{
	mLevelCollider.push_back(_c);
	mLevelCollider[mLevelCollider.size() - 1].recalculateMinMax();

	mDirtyColliders = true;
}

void Level::setReady(bool _b)
{
	assert("Zero particles to spawn!" && mNumberOfSpawnParticles > 0);
	assert("No particle system set!" && mExternalParticleSystem != nullptr);

	mIsReady = _b;
}

bool Level::isReady()
{
	return mIsReady;
}

bool Level::isRunning()
{
	return mIsRunning;
}

bool Level::isPaused()
{
	return mIsPaused;
}

uint Level::getScore()
{
	return mCurrentScore;
}

float Level::getCurrentTime()
{
	return mCurrentTimeSec;
}

void Level::setParticleSystem(ParticleSystem * _ps)
{
	mExternalParticleSystem = _ps;
}

void Level::setEndzone(ofVec3f _position, ofVec3f _size)
{
	mEndzone.setPosition(_position);
	mEndzone.set(_size.x, _size.y, _size.z);
	mEndzone.recalculateMinMax();
}

void Level::setStartzone(ofVec3f _position, ofVec3f _size, uint _numberOfSpawnParticles)
{
	mStartzone.setPosition(_position);
	mStartzone.set(_size.x, _size.y, _size.z);
	mNumberOfSpawnParticles = _numberOfSpawnParticles;
}

void Level::setLevelShader(ofShader * _s)
{
	mLevelShader = _s;
}

void Level::setSunDirection(ofVec3f * _sd)
{
	mSunDirection = _sd;
}

const std::vector<Cube>& Level::getLevelColliders() const
{
	return mLevelCollider;
}

void Level::resetLevel()
{
	mCurrentScore = 0;
	mIsRunning = false;
	mIsPaused = false;
	mExternalParticleSystem->setNumberOfParticles(0);
}

void Level::draw(ofVec3f& _cameraPos, ofPolyRenderMode _rm)
{
	mLevelShader->begin();
	mLevelShader->setUniform3f("systemPos", ofVec3f(0.f));
	mLevelShader->setUniform1i("endZone", 0);
	mLevelShader->setUniform3f("cameraPos", _cameraPos);
	mLevelShader->setUniform3f("sunDir", *mSunDirection);

	for (size_t i = 0; i < mLevelCollider.size(); ++i)
	{
		mLevelCollider[i].draw(_rm);
	}

	mLevelShader->setUniform1i("endZone", 1);
	mEndzone.draw();

	mLevelShader->end();
}

void Level::update(float dt)
{
	if (!this->isRunning())
		return;

	if (mDirtyColliders)
	{
		std::vector<MinMaxData> minMax(mLevelCollider.size());
		for (size_t i = 0; i < mLevelCollider.size(); i++)
		{
			minMax[i] = mLevelCollider[i].getLocalMinMax() + mLevelCollider[i].getPosition();
		}
		mExternalParticleSystem->setStaticCollision(minMax);

		mDirtyColliders = false;
	}

	mCurrentTimeSec += dt;
	if (mCurrentTimeSec >= mSecondsToFinishLevel)
	{
		this->end();
	}
}

void Level::start()
{
	if (mIsRunning == false && isReady())
	{
		mIsRunning = true;
		ofVec3f tmpSize = mStartzone.getSize();
		//mParticleSystem->addCube(tmpSize * ofVec3f(ofRandom(1.0f), 1, ofRandom(1.0f)), tmpSize, mXmlSettings.getValue("GENERAL:DROPSIZE", 1000));
		mExternalParticleSystem->addCube(ofVec3f(0, 0.5f, 0), tmpSize, mNumberOfSpawnParticles);
		
	}
}

void Level::end()
{
	mIsRunning = false;
	mIsPaused = false;
}

void Level::pause(bool _p)
{
	mIsPaused = _p;
}

bool Endzone::isActive()
{
	return mIsActive;
}

bool Endzone::setActive(bool _b)
{
	mIsActive = _b;
	return mIsActive;
}
