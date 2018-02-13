#pragma once

// Standardized MAX, MIN and CLAMP
//#define MAX(a, b) ((a > b) ? a : b)
//#define MIN(a, b) ((a < b) ? a : b)
//#define CLAMP(a, b, c) MIN(MAX(a, b), c)    // double sided clip of input a
//#define TOPCLAMP(a, b) (a < b ? a:b)	    // single top side clip of input a

#include <ofVec3f.h>
#include <ofRectangle.h>
#include <ofMath.h>
#include <ofQuaternion.h>
#include <ofShader.h>
#include <ofVbo.h>

#include "oclHelper.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>
#include <helper_math.h>

//general definitions
typedef unsigned int uint;

struct SimulationData
{
	float smoothingWidth;
};

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
	const float smoothingWidth,
	const float4 gravity,
	const float4 dimension,
	const uint numberOfParticles);

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
	enum class ComputeMode
	{
		CPU = 0,
		COMPUTE_SHADER,
		CUDA,
		OPENCL,
		COMPUTEMODES_SIZE
	};

	ParticleSystem(uint mp);
	~ParticleSystem();

	void setDimensions(ofVec3f dimensions);
	void setNumberOfParticles(uint nop);
	void setRotation(ofQuaternion rotation);
	void setMode(ComputeMode m);
	ComputeMode nextMode(ParticleSystem::ComputeMode current);
	void setSmoothingWidth(float sw);
	void addDamBreak(uint particleAmount);
	void addRandom(uint particleAmount);
	void addCube(ofVec3f position, ofVec3f size, uint particleAmount);
	void addDrop();
	void draw();
	ofVec4f* getPositionPtr();
	ofVec3f getDimensions();
	uint getNumberOfParticles();
	uint getCapacity();
	ComputeMode getMode();
	CUDAta& getCudata();

	void update(float dt);
	uint debug_testIfParticlesOutside();

private:
	uint mNumberOfParticles,
		mCapacity;
	ComputeMode mMode;
	std::unordered_map<ComputeMode, bool> mAvailableModes;
	ofVec4f	*mPosition,
		*mVelocity;
	ofVec3f	mDimension,
		mGravity,
		mGravityRotated;
	ofVbo mParticlesVBO;
	ofQuaternion mRotation;
	ComputeShaderData mComputeData;
	SimulationData mSimData;
	oclHelper mOCLHelper;
	OCLData mOCLData;
	CUDAta mCUData;

	void iUpdateCPU(float dt);
	void iUpdateCompute(float dt);
	void iUpdateOCL(float dt);
	void iUpdateCUDA(float dt);
	ofVec3f iCalculatePressureVector(size_t index);
};

