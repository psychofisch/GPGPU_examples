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
#include "Stopwatch.h"

#ifndef HANDLE_GL_ERROR()
#define HANDLE_GL_ERROR() {GLenum err; while ((err = glGetError()) != GL_NO_ERROR) ofLogWarning() << __FILE__ << ":" << __LINE__ << ": GL error = " << err;}
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

	// returns the current mode as enum
	ComputeMode getMode() const;
	// returns the current mode as std::string
	std::string getModeAsString() const;
	// returns the next available mode but does not set it
	ComputeMode nextMode(CollisionSystem::ComputeMode current) const;
	// sets a new computation mode; usually used in combination with "nextMode"
	void setMode(ComputeMode m);
	// measures the next calculation
	void measureNextCalculation();
	// gets the result of the last measurement
	double getMeasurement() const;
	// method that calculates and returns the collisions
	void getCollisions(const std::vector<Cube>& cubes, OUT std::vector<int>& collisions);
	
	// statics

	// converts the ComputeMode enum to string
	static std::string getComputeModeString(ComputeMode m);

private:
	ComputeMode mMode;
	bool mAvailableModes[static_cast<size_t>(ComputeMode::COMPUTEMODES_SIZE)];
	ComputeShaderData mComputeData;
	CUDAta mCudata;
	oclHelper mOCLHelper;
	OCLData mOCLData;
	ThrustHelper::ThrustData mThrustData;
	bool mCPUThreshold,
		mMeasureNext = false;
	double mLastMeasurement;
	Stopwatch mClock;

	void iGetCollisionsCPU(std::vector<MinMaxData>& cubes, OUT std::vector<int>& collisions);
	void iGetCollisionsCompute(std::vector<MinMaxData>& cubes, OUT std::vector<int>& collisions);
	void iGetCollisionsCUDA(std::vector<MinMaxData>& cubes, OUT std::vector<int>& collisions);
	void iGetCollisionsOCL(std::vector<MinMaxData>& cubes, OUT std::vector<int>& collisions);
	void iGetCollisionsThrust(std::vector<MinMaxData>& cubes, OUT std::vector<int>& collisions);
};
