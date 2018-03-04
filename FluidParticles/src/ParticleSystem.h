#pragma once

// Standardized MAX, MIN and CLAMP
//#define MAX(a, b) ((a > b) ? a : b)
//#define MIN(a, b) ((a < b) ? a : b)
//#define CLAMP(a, b, c) MIN(MAX(a, b), c)    // double sided clip of input a
//#define TOPCLAMP(a, b) (a < b ? a:b)	    // single top side clip of input a

// Openframeworks includes
#include <ofVec3f.h>
#include <ofRectangle.h>
#include <ofMath.h>
#include <ofQuaternion.h>
#include <ofShader.h>
#include <ofVbo.h>
#include <ofxXmlSettings.h>

// CUDA includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>
#include <helper_math.h>

// own includes
#include "oclHelper.h"
#include "Stopwatch.h"
#include "ParticleDefinitions.h"

//general definitions
//#ifdef _WIN32
//
//#include <intrin.h>
//uint64_t rdtsc() {
//	return __rdtsc();
//}
//
//#endif

//definitions for Compute Shader
struct ComputeShaderData
{
	ofShader computeShader;
	ofBufferObject positionBuffer;
	ofBufferObject positionOutBuffer;
	ofBufferObject velocityBuffer;
};

//definitions for OpenCL
struct OCLData
{
	size_t maxWorkGroupSize;
	cl::Buffer positionBuffer;
	cl::Buffer positionOutBuffer;
	cl::Buffer velocityBuffer;
};

//definitions for CUDA
extern "C" void cudaUpdate(
	float4* position,
	float4* positionOut,
	float4* velocity,
	const float dt,
	const float3 gravity,
	const float3 dimension,
	const uint numberOfParticles,
	SimulationData simData);

struct CUDAta
{
	size_t maxWorkGroupSize;
	float4 *position;
	float4 *positionOut;
	float4 *velocity;

	struct cudaGraphicsResource *cuPos;
	struct cudaGraphicsResource *cuPosOut;
	struct cudaGraphicsResource *cuVel;
};

//class definition
class ParticleSystem
{
public:
	enum ComputeMode
	{
		CPU = 0,
		COMPUTE_SHADER,
		CUDA,
		OPENCL,
		COMPUTEMODES_SIZE // these values are used as array indices, dont delete this!
	};

	ParticleSystem(uint mp);
	~ParticleSystem();

	void setupAll(ofxXmlSettings& settings);
	void setupCPU(ofxXmlSettings& settings);
	void setupCompute(ofxXmlSettings& settings);
	void setupCUDA(ofxXmlSettings& settings);
	void setupOCL(ofxXmlSettings& settings);

	void setDimensions(ofVec3f dimensions);
	void setNumberOfParticles(uint nop);
	void setRotation(ofQuaternion rotation);
	void setMode(ComputeMode m);
	ComputeMode nextMode(ParticleSystem::ComputeMode current) const;
	void addDamBreak(uint particleAmount);
	void addCube(ofVec3f position, ofVec3f size, uint particleAmount, bool random = false);
	void draw();
	ofVec3f getDimensions();
	uint getNumberOfParticles();
	uint getCapacity();
	ComputeMode getMode();
	CUDAta& getCudata();

	//Simulation Data
	void setSimulationData(SimulationData& sim);
	void setSmoothingWidth(float sw);
	void setRestDensity(float rd);

	void measureNextUpdate();

	void update(float dt);
	uint debug_testIfParticlesOutside();

private:
	uint mNumberOfParticles,
		mCapacity,
		mThreshold;
	ComputeMode mMode;
	bool mAvailableModes[static_cast<size_t>(ComputeMode::COMPUTEMODES_SIZE)];
	ofVec4f	*mPosition,
		*mVelocity;
	ofVec3f	mDimension,
		mGravity;
	ofVbo mParticlesVBO;
	ofBufferObject mParticlesBuffer;
	ofQuaternion mRotation;
	ComputeShaderData mComputeData;
	SimulationData mSimData;
	oclHelper mOCLHelper;
	OCLData mOCLData;
	CUDAta mCUData;
	Stopwatch mClock;
	bool mMeasureTime;

	void iUpdateCPU(float dt);
	void iUpdateCompute(float dt);
	void iUpdateOCL(float dt);
	void iUpdateCUDA(float dt);
	ofVec3f iCalculatePressureVector(size_t index, ofVec4f pos, ofVec4f vel);
};

