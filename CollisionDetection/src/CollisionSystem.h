#pragma once

#include <ofxXmlSettings.h>

#include "Cube.h"

// general definitions
struct MinMaxData
{
	ofVec3f min, max;
};

//definitions for Compute Shader
struct ComputeShaderData
{
	ofShader computeShader;
	ofBufferObject positionBuffer;
	ofBufferObject positionOutBuffer;
	ofBufferObject velocityBuffer;
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

	void getCollisions(std::vector<Cube>& cubes, OUT std::vector<int>& collisions);
private:
	ComputeMode mMode;
	bool mAvailableModes[static_cast<size_t>(ComputeMode::COMPUTEMODES_SIZE)];
	//std::vector<MinMaxData> mMinMax;

	void iGetCollisionsCPU(std::vector<Cube>& cubes, OUT std::vector<int>& collisions);
};
