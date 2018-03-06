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

//Thrust includes
#include <thrust\device_vector.h>
#include <thrust\host_vector.h>
#include <thrust\device_malloc.h>
#include <thrust\copy.h>

// own includes
#include "oclHelper.h"
#include "Stopwatch.h"
#include "ParticleDefinitions.h"
#include "ThrustHelper.h"

//general definitions
#ifdef _DEBUG
	#define CUDAERRORS(x) checkCudaErrors(x)
#else
	#define CUDAERRORS(x) x
#endif
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
};

//definitions for Thrust
struct ThrustData
{
	thrust::device_ptr<float4> position;
	thrust::device_ptr<float4> positionOut;
	thrust::device_ptr<float4> velocity;
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
		THRUST,
		COMPUTEMODES_SIZE // these values are used as array indices, dont delete this!
	};

	ParticleSystem(uint mp);
	~ParticleSystem();

	// setup methods
	void setupAll(ofxXmlSettings& settings);
	void setupCPU(ofxXmlSettings& settings);
	void setupCompute(ofxXmlSettings& settings);
	void setupCUDA(ofxXmlSettings& settings);
	void setupOCL(ofxXmlSettings& settings);
	void setupThrust(ofxXmlSettings& settings);

	// simple getters

	ofVec3f getDimensions() const;
	uint getNumberOfParticles() const;
	uint getCapacity() const;
	ComputeMode getMode() const;
	CUDAta& getCudata();

	// simple setters

	// sets the dimension of the bounding volume
	void setDimensions(ofVec3f dimensions);
	// manually set the number of particles; mainly used to clear the particle system
	void setNumberOfParticles(uint nop);
	// *WIP* rotates the gravity vector
	void setRotation(ofQuaternion rotation);
	// sets a new computation mode; usually used in combination with "nextMode"
	void setMode(ComputeMode m);

	// other methods

	//only returns the next available mode but does not set it
	ComputeMode nextMode(ParticleSystem::ComputeMode current) const;
	// adds a traditional dam (just a convience function, addCube can achieve the same)
	void addDamBreak(uint particleAmount);
	// adds the given number of particles in a cube form 
	void addCube(ofVec3f position, ofVec3f size, uint particleAmount, bool random = false);
	// draws the particle VBO
	void draw();

	//Simulation

	// sets the all parameters that are needed for the simulation
	void setSimulationData(SimulationData& sim);
	// calculates the next step in the simulation
	void update(float dt);

	// DEBUG

	//sets a flag that will trigger a clock to measure the duration of the next update
	void measureNextUpdate();
	// returns the number of particles that are outside of the bounding volume
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
	ThrustData mThrustData;
	Stopwatch mClock;
	bool mMeasureTime;

	void iUpdateCPU(float dt);
	void iUpdateCompute(float dt);
	void iUpdateOCL(float dt);
	void iUpdateCUDA(float dt);
	void iUpdateThrust(float dt);
	ofVec3f iCalculatePressureVector(size_t index, ofVec4f pos, ofVec4f vel);
};

