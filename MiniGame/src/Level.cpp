#include "Level.h"

Level::Level()
	:mExternalParticleSystem(nullptr),
	mCurrentScore(0),
	mState(GameState::UnReady)
	//mIsReady(false),
	//mIsRunning(false)
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
	assert("No end time set!" && mSecondsToFinishLevel > 0.f);

	mState = GameState::Ready;
	//mIsReady = _b;
}

bool Level::isReady() const
{
	if (mState == GameState::Ready)
		return true;
	else
		return false;
}

bool Level::isRunning() const
{
	if (mState == GameState::Running)
		return true;
	else
		return false;
}

bool Level::isPaused() const
{
	if (mState == GameState::Paused)
		return true;
	else
		return false;
}

uint Level::getScore() const
{
	return mCurrentScore;
}

uint Level::getSpawnedParticles() const
{
	return mNumberOfSpawnParticles;
}

float Level::getCurrentTime() const
{
	return mCurrentTimeSec;
}

float Level::getLevelTime() const
{
	return mSecondsToFinishLevel;
}

Level::GameState Level::getGameState() const
{
	return mState;
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

void Level::setTimeToFinish(float _seconds)
{
	mSecondsToFinishLevel = _seconds;
}

const std::vector<Cube>& Level::getLevelColliders() const
{
	return mLevelCollider;
}

void Level::resetLevel()
{
	mCurrentScore = 0;
	mCurrentTimeSec = 0.f;
	mExternalParticleSystem->setNumberOfParticles(0);
	mState = GameState::Ready;
}

void Level::draw(ofVec3f& _cameraPos, ofPolyRenderMode _rm)
{
	mLevelShader->begin();
	mLevelShader->setUniform3f("systemPos", ofVec3f(0.f));
	mLevelShader->setUniform1i("endZone", 0);
	mLevelShader->setUniform3f("cameraPos", _cameraPos);
	mLevelShader->setUniform3f("sunDir", *mSunDirection);
	mLevelShader->setUniform1f("alpha", 1.f);
	mLevelShader->setUniform1i("world", 1);
	mLevelShader->setUniform4f("objColor", ofFloatColor::aliceBlue);

	for (size_t i = 0; i < mLevelCollider.size(); ++i)
	{
		mLevelCollider[i].draw(_rm);
	}

	mLevelShader->setUniform4f("objColor", ofFloatColor::red);
	mLevelShader->setUniform1f("alpha", 0.5f);
	mEndzone.draw();

	mLevelShader->end();
}

void Level::update(float dt)
{
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

	if (!this->isRunning())
		return;

	if (mExternalParticleSystem->getNumberOfParticles() > 0)
	{
		uint removed = mExternalParticleSystem->removeInVolume(mEndzone.getGlobalMinMax());
		mCurrentScore += removed;
	}
	else
	{
		mState = GameState::Finished;
		this->end();
	}

	mCurrentTimeSec += dt;
	if (mCurrentTimeSec >= mSecondsToFinishLevel)
	{
		mState = GameState::TimeOver;
		this->end();
	}
}

void Level::start()
{
	if (mState == GameState::Ready && isReady())
	{
		mState = GameState::Running;
		ofVec3f tmpSize = mStartzone.getSize();
		//mParticleSystem->addCube(tmpSize * ofVec3f(ofRandom(1.0f), 1, ofRandom(1.0f)), tmpSize, mXmlSettings.getValue("GENERAL:DROPSIZE", 1000));
		mExternalParticleSystem->addCube(mStartzone.getPosition(), mStartzone.getSize(), mNumberOfSpawnParticles);
		
	}
}

void Level::end()
{
	
}

void Level::pause(bool _p)
{
	if (mState == GameState::Running && _p == true)
		mState = GameState::Paused;
	else if(mState == GameState::Paused && _p == false)
		mState = GameState::Running;
}

std::string Level::convertGamestateToString(Level::GameState _state)
{
	std::string text;

	switch (_state)
	{
	case Level::UnReady:text = "Unready";
		break;
	case Level::Ready:text = "Ready";
		break;
	case Level::Running:text = "Running";
		break;
	case Level::Paused:text = "Paused";
		break;
	case Level::Finished:text = "Finished";
		break;
	case Level::TimeOver:text = "TimeOver";
		break;
	default:text = "Unknown";
		break;
	}

	return text;
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
