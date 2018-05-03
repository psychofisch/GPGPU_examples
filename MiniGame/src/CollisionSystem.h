#pragma once

// general includes
#include <ofxXmlSettings.h>

// CUDA includes
#include <cuda_runtime.h>
#include <helper_cuda.h>

// OpenCL includes
#include "oclHelper.h"

// own includes
#include "CollisionDefinitions.h"
#include "Cube.h"
#include "ThrustHelper.h"

namespace Collision
{
	// general definitions
#ifdef _DEBUG
#define GLERROR std::cout << __LINE__ << ": " << glGetError() << std::endl
#else
#define GLERROR
#endif

//definitions for Compute Shader
	struct ComputeShaderData
	{
		ofShader computeShader;
		ofBufferObject minMaxBuffer;
		ofBufferObject collisionBuffer;
	};

	//definitions for CUDA
	extern "C" void cudaGetCollisions(
		float4* minMaxBuffer,
		int* collisionBuffer,
		const int amountOfCubes);

	struct CUDAta
	{
		size_t maxWorkGroupSize;
		size_t currentArraySize;
		float4 *minMaxBuffer;
		int *collisionBuffer;
	};

	struct OCLData
	{
		size_t maxWorkGroupSize;
		size_t currentArraySize;
		cl::Buffer minMaxBuffer;
		cl::Buffer collisionBuffer;
	};

	float4 make_float4(ofVec4f v_);
	float3 make_float3(ofVec4f v_);
}

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
	Collision::ComputeShaderData mComputeData;
	Collision::CUDAta mCudata;
	oclHelper mOCLHelper;
	Collision::OCLData mOCLData;
	ThrustHelper::ThrustCollisionData mThrustData;
	bool mCPUThreshold;

	void iGetCollisionsCPU(std::vector<Cube>& cubes, OUT std::vector<int>& collisions);
	void iGetCollisionsCompute(std::vector<Cube>& cubes, OUT std::vector<int>& collisions);
	void iGetCollisionsCUDA(std::vector<Cube>& cubes, OUT std::vector<int>& collisions);
	void iGetCollisionsOCL(std::vector<Cube>& cubes, OUT std::vector<int>& collisions);
	void iGetCollisionsThrust(std::vector<Cube>& cubes, OUT std::vector<int>& collisions);
};