#pragma once

// Openframeworks includes
#include <ofVec3f.h>
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
#include "CollisionDefinitions.h"
#include "CudaHelper.h"

namespace Particle
{
	//general definitions
#ifdef _DEBUG
#define CUDAERRORS(x) checkCudaErrors(x)
#define HANDLE_GL_ERROR() {GLenum err; while ((err = glGetError()) != GL_NO_ERROR) ofLogNotice() << __FILE__ << ":" << __LINE__ << ": GL error:	" << err;}
#else
#define CUDAERRORS(x) x
#define HANDLE_GL_ERROR()
#endif

	const ofVec3f directions[6] = { ofVec3f(1.f, 0, 0),
									ofVec3f(-1.f, 0, 0),
									ofVec3f(0, 1.f, 0),
									ofVec3f(0, -1.f, 0),
									ofVec3f(0, 0, 1.f),
									ofVec3f(0, 0, -1.f) };
	const ofVec3f gravity = ofVec3f(0, -9.81f, 0);

	//definitions for Compute Shader
	struct ComputeShaderData
	{
		ofShader computeShader;
		ofBufferObject positionBuffer;
		ofBufferObject positionOutBuffer;
		ofBufferObject velocityBuffer;
		ofBufferObject staticCollisionBuffer;
		size_t workGroupCount;
	};

	//definitions for OpenCL
	struct OCLData
	{
		size_t maxWorkGroupSize;
		cl::Buffer positionBuffer;
		cl::Buffer positionOutBuffer;
		cl::Buffer velocityBuffer;
		cl::Buffer staticCollisionBuffer;
		size_t allocatedColliders;
	};

	//definitions for CUDA
	struct CUDAta
	{
		size_t maxWorkGroupSize;
		float4 *position;
		float4 *positionOut;
		float4 *velocity;
		MinMaxData *staticCollisionBuffer;
		size_t allocatedColliders;
	};
}

// external CUDA function in the particles.cu file
extern "C" void cudaParticleUpdate(
	float4* positions,
	float4* positionOut,
	float4* velocity,
	MinMaxData* staticColliders,
	const float dt,
	const float3 gravity,
	const float3 position,
	const float3 dimension,
	const size_t numberOfParticles,
	const size_t numberOfColliders,
	SimulationData simData);

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
	void createParticleShader(std::string vert, std::string frag);

	// simple getters

	ofVec3f getDimensions() const;
	uint getNumberOfParticles() const;
	uint getCapacity() const;
	ComputeMode getMode() const;
	Particle::CUDAta& getCudata();
	ofVec3f getPosition() const;
	ofVec3f getGravity() const;

	// simple setters

	// sets the dimension of the bounding volume
	void setDimensions(ofVec3f dimensions);
	// manually set the number of particles; mainly used to clear the particle system
	void setNumberOfParticles(uint nop);
	// *WIP* rotates the gravity vector
	void setRotation(ofQuaternion rotation);
	// sets a new computation mode; usually used in combination with "nextMode"
	void setMode(ComputeMode m);
	// sets the position of the particle system
	void setPosition(ofVec3f p);
	// adds the given value to the current position of the particle system
	void addPosition(ofVec3f p);
	// set static collisions
	void setStaticCollision(std::vector<MinMaxData>& collision);
	// set gravity
	void setGravity(ofVec3f g);

	// other methods

	//only returns the next available mode but does not set it
	ComputeMode nextMode(ParticleSystem::ComputeMode current) const;
	// adds a traditional dam (just a convience function, addCube can achieve the same)
	void addDamBreak(uint particleAmount);
	// adds the given number of particles in a cube form 
	void addCube(ofVec3f position, ofVec3f size, uint particleAmount, bool random = false);
	// draws the particle VBO
	void draw(const ofVec3f& _camera, const ofVec3f& _sunDir, ofPolyRenderMode _rm);
	// removes all particles that are in the endzone and returns how many particles got removed
	uint removeInVolume(MinMaxData v);

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
	// a generic switch used in debugging
	void toggleGenericSwitch();

private:
	size_t mNumberOfParticles,
		mCapacity,
		mThreshold;
	ComputeMode mMode;
	ofVboMesh mParticleModel;
	ofSpherePrimitive mParticleTmp;
	ofShader mParticleShader;
	std::array<bool, static_cast<size_t>(ComputeMode::COMPUTEMODES_SIZE)> mAvailableModes;
	std::vector<ofVec4f> mParticlePosition,
		mParticleVelocity;
	ofVec3f	mDimension,
		mGravity,
		mPosition;
	ofVbo mParticlesVBO;
	ofBufferObject mParticlesBuffer;
	ofQuaternion mRotation;
	Particle::ComputeShaderData mComputeData;
	SimulationData mSimData;
	oclHelper mOCLHelper;
	Particle::OCLData mOCLData;
	Particle::CUDAta mCUData;
	std::unique_ptr<ThrustHelper::ThrustParticleData> mThrustData;
	Stopwatch mClock;
	std::vector<MinMaxData> mStaticCollision;
	bool mMeasureTime,
		mGenericSwitch;

	void iUpdateCPU(float dt);
	void iUpdateCompute(float dt);
	void iUpdateOCL(float dt);
	void iUpdateCUDA(float dt);
	void iUpdateThrust(float dt);
	ofVec3f iCalculatePressureVector(size_t index, ofVec4f pos, ofVec4f vel, float dt);
};

