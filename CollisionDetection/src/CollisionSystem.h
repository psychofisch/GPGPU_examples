#pragma once

#include <ofxXmlSettings.h>

#include "Cube.h"

#ifdef _DEBUG
#define GLERROR std::cout << __LINE__ << ": " << glGetError() << std::endl
#else
#define GLERROR
#endif

// general definitions
struct MinMaxData
{
	ofVec3f min, max;
};

//definitions for Compute Shader
struct ComputeShaderData
{
	ofShader computeShader;
	ofBufferObject minMaxBuffer;
	ofBufferObject collisionBuffer;
};

class CollisionSystem
{
public:
	enum ComputeMode
	{
		CPU = 0,
		COMPUTE_SHADER,
		CUDA,
		OPENCL,
		THRUST,
		COMPUTEMODES_SIZE // these values are used as array indices, dont delete this!
	};

	CollisionSystem();
	~CollisionSystem();

	// setup methods
	void setupAll(ofxXmlSettings& settings);
	void setupCPU(ofxXmlSettings& settings);
	void setupCompute(ofxXmlSettings& settings);
	void setupCUDA(ofxXmlSettings& settings);
	void setupOCL(ofxXmlSettings& settings);
	void setupThrust(ofxXmlSettings& settings);

	ComputeMode getMode() const;
	//only returns the next available mode but does not set it
	ComputeMode nextMode(CollisionSystem::ComputeMode current) const;
	// sets a new computation mode; usually used in combination with "nextMode"
	void setMode(ComputeMode m);

	void getCollisions(std::vector<Cube>& cubes, OUT std::vector<int>& collisions);
private:
	ComputeMode mMode;
	bool mAvailableModes[static_cast<size_t>(ComputeMode::COMPUTEMODES_SIZE)];
	//std::vector<MinMaxData> mMinMax;
	ComputeShaderData mComputeData;

	void iGetCollisionsCPU(std::vector<Cube>& cubes, OUT std::vector<int>& collisions);
	void iGetCollisionsCompute(std::vector<Cube>& cubes, OUT std::vector<int>& collisions);
};
