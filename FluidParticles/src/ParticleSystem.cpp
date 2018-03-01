#include "ParticleSystem.h"

ParticleSystem::ParticleSystem(uint maxParticles)
	:mGravity(0.f, -9.81f, 0.f),
	mCapacity(maxParticles),
	mNumberOfParticles(0),
	mMode(ComputeMode::CPU)
{
	//*** general setup
	// these buffers are mandatory, even when CPU mode is not enabled
	// they are used as cache between different GPU modes
	mPosition = new ofVec4f[maxParticles];
	mVelocity = new ofVec4f[maxParticles];
	//***

	//*** general GL setup
	// use compute shader buffer to store the particle position on the GPU
	mComputeData.positionOutBuffer.allocate(sizeof(ofVec4f) * mCapacity, mPosition, GL_DYNAMIC_DRAW);
	mComputeData.positionBuffer.allocate(sizeof(ofVec4f) * mCapacity, mPosition, GL_DYNAMIC_DRAW);
	mComputeData.velocityBuffer.allocate(sizeof(ofVec4f) * mCapacity, mVelocity, GL_DYNAMIC_DRAW);

	mParticlesVBO.setVertexBuffer(mComputeData.positionBuffer, 3, sizeof(ofVec4f));
}

ParticleSystem::~ParticleSystem()
{
	delete[] mPosition;
	delete[] mVelocity;
}

void ParticleSystem::setupAll(ofxXmlSettings & settings)
{
	setupCPU(settings);
	setupCompute(settings);
	setupCUDA(settings);
	setupOCL(settings);
}

void ParticleSystem::setupCPU(ofxXmlSettings & settings)
{
	mThreshold = settings.getValue("CPU:THRESHOLD", 1000);

	mAvailableModes[ComputeMode::CPU] = settings.getValue("CPU:ENABLED", true);
}

void ParticleSystem::setupCompute(ofxXmlSettings & settings)
{
	// the Compute Shader uses the OpenGL buffers, so there's no need to allocate additional memory
	if (mComputeData.computeShader.setupShaderFromFile(GL_COMPUTE_SHADER, settings.getValue("COMPUTE:SOURCE", "particles.compute"))
		&& mComputeData.computeShader.linkProgram())
		mAvailableModes[ComputeMode::COMPUTE_SHADER] = settings.getValue("COMPUTE:ENABLED", true);
	else
		mAvailableModes[ComputeMode::COMPUTE_SHADER] = false;
}

void ParticleSystem::setupCUDA(ofxXmlSettings & settings)
{
	//load CUDA command line arguments from settings file
	const int cmdArgc = settings.getValue("CUDA:ARGC", 0);
	const char* cmdArgs = settings.getValue("CUDA:ARGV", "").c_str();
	
	// find a CUDA device
	findCudaDevice(cmdArgc, &cmdArgs);

	// register all the GL buffers to CUDA
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&mCUData.cuPos, mComputeData.positionBuffer.getId(), cudaGraphicsMapFlagsNone));//TODO: change flags
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&mCUData.cuPosOut, mComputeData.positionOutBuffer.getId(), cudaGraphicsMapFlagsNone));
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&mCUData.cuVel, mComputeData.velocityBuffer.getId(), cudaGraphicsMapFlagsNone));
	
	// "checkCudaErrors" will quit the program in case of a problem, so it is safe to assume that if the program reached this point CUDA will work
	mAvailableModes[ComputeMode::CUDA] = settings.getValue("CUDA:ENABLED", true);
}

void ParticleSystem::setupOCL(ofxXmlSettings & settings)
{
	int platformID = settings.getValue("OCL:PLATFORMID", 0);
	int deviceID = settings.getValue("OCL:DEVICEID", 0);
	std::string sourceFile = settings.getValue("OCL:SOURCE", "data/particles.cl");

	// try to set up an OpenCL context
	if (!mOCLHelper.setupOpenCLContext(platformID, deviceID))
	{
		// try to compile the source file
		if (!mOCLHelper.compileKernel(sourceFile.c_str()))
		{
			// if all of the above worked -> set up all buffers and settings
			cl::Context context = mOCLHelper.getCLContext();

			// create buffers on the OpenCL device (don't need to be the GPU)
			mOCLData.positionBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(ofVec4f) * mCapacity);
			mOCLData.positionOutBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(ofVec4f) * mCapacity);
			mOCLData.velocityBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(ofVec4f) * mCapacity);

			// query the maximum work group size rom the device
			cl_int err = mOCLHelper.getDevice().getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &mOCLData.maxWorkGroupSize);
			mOCLData.maxWorkGroupSize /= 2;//testing showed best performance at half the max ThreadCount
			oclHelper::handle_clerror(err, __LINE__);

			mAvailableModes[ComputeMode::OPENCL] = settings.getValue("OCL:ENABLED", true);
		}
		else
		{
			std::cout << "ERROR: Unable to compile \"" << sourceFile << "\".\n";
			mAvailableModes[ComputeMode::OPENCL] = false;
		}
	}
	else
	{
		std::cout << "ERROR: Unable to create OpenCL context\n";
		mAvailableModes[ComputeMode::OPENCL] = false;
	}
}

// mainly used to clear the particle system
void ParticleSystem::setNumberOfParticles(uint nop)
{
	if (nop > mCapacity || mCapacity == 0)
	{
		return;
	}

	mNumberOfParticles = nop;
}

void ParticleSystem::setDimensions(ofVec3f dimensions)
{
	mDimension = dimensions;
}

void ParticleSystem::setRotation(ofQuaternion rotation)
{
	mRotation = rotation;
}

void ParticleSystem::setMode(ComputeMode m)
{
	// sync all data back to RAM
	if (mMode == ComputeMode::COMPUTE_SHADER || mMode == ComputeMode::CUDA)
	{
		ofVec4f* tmpPtrFromGPU = mComputeData.positionBuffer.map<ofVec4f>(GL_READ_ONLY);
		std::copy(tmpPtrFromGPU, tmpPtrFromGPU + mNumberOfParticles, mPosition);
		mComputeData.positionBuffer.unmap();

		tmpPtrFromGPU = mComputeData.velocityBuffer.map<ofVec4f>(GL_READ_ONLY);
		std::copy(tmpPtrFromGPU, tmpPtrFromGPU + mNumberOfParticles, mVelocity);
		mComputeData.velocityBuffer.unmap();
	}
	else if (mMode == ComputeMode::OPENCL)
	{
		mOCLHelper.getCommandQueue().enqueueReadBuffer(mOCLData.positionOutBuffer, CL_TRUE, 0, mNumberOfParticles * sizeof(ofVec4f), mPosition);
		mOCLHelper.getCommandQueue().enqueueReadBuffer(mOCLData.velocityBuffer, CL_TRUE, 0, mNumberOfParticles * sizeof(ofVec4f), mVelocity);
	}
	/*else if (m == ComputeMode::CUDA)// keep this, just in case
	{
		cudaDeviceSynchronize();
		memcpy(mPosition, mCUData.position, sizeof(ofVec4f) * mNumberOfParticles);
	}*/

	// copy the data to the corresponding buffer for the new mode
	if (m == ComputeMode::COMPUTE_SHADER || m == ComputeMode::CUDA)
	{
		mComputeData.positionBuffer.updateData(sizeof(ofVec4f) * mNumberOfParticles, mPosition);
		mComputeData.velocityBuffer.updateData(sizeof(ofVec4f) * mNumberOfParticles, mVelocity);
	}
	else if (m == ComputeMode::OPENCL)
	{
		mOCLHelper.getCommandQueue().enqueueWriteBuffer(mOCLData.positionBuffer, CL_TRUE, 0, mNumberOfParticles * sizeof(ofVec4f), mPosition);
		mOCLHelper.getCommandQueue().enqueueWriteBuffer(mOCLData.velocityBuffer, CL_TRUE, 0, mNumberOfParticles * sizeof(ofVec4f), mVelocity);
	}
	/*else if (m == ComputeMode::CUDA)// keep this, just in case
	{
		memcpy(mCUData.position, mPosition, sizeof(ofVec4f) * mNumberOfParticles);
		cudaDeviceSynchronize();
	}*/

	// set mode
	mMode = m;
}

// returns the next available mode
// IMPORTANT: does not set anything
ParticleSystem::ComputeMode ParticleSystem::nextMode(ParticleSystem::ComputeMode current) const
{
	int enumSize = ComputeMode::COMPUTEMODES_SIZE;

	for (int i = 0; i < enumSize; ++i)
	{
		int next = (current + 1 + i) % enumSize;
		if (mAvailableModes[next] // checks if the next mode is even available
			&& !(next == ComputeMode::CPU && mNumberOfParticles > mThreshold)) // checks if there are more particles than the CPU threshold allows
			return static_cast<ComputeMode>(next);
	}

	return current;
}

void ParticleSystem::setSimulationData(SimulationData & sim)
{
	mSimData = sim;
}

void ParticleSystem::setSmoothingWidth(float sw)
{
	mSimData.interactionRadius = sw;
}

void ParticleSystem::setRestDensity(float rd)
{
	mSimData.rho0 = rd;
}

// helper function to create a dam break
void ParticleSystem::addDamBreak(uint particleAmount)
{
	ofVec3f damSize = mDimension;
	damSize.x *= 0.2f;
	addCube(ofVec3f(0) + 0.1f, damSize - 0.1f, particleAmount);
}

// adds the given number of particles in a cube form
void ParticleSystem::addCube(ofVec3f cubePos, ofVec3f cubeSize, uint particleAmount, bool random)
{
	//some calculations to distribute the particles evenly
	float cubedParticles = powf(particleAmount, 1.f / 3.f) * 3;
	float ratio;
	ratio = cubeSize.x / (cubeSize.x + cubeSize.y + cubeSize.z);
	float gap;
	gap = cubeSize.x / (cubedParticles * ratio);

	ofVec3f partPos(0.f);
	uint particleCap = -1;

	for (uint i = 0; i < particleAmount; i++)
	{
		// stops if the maximum number of particles is reached
		if (mNumberOfParticles + i >= mCapacity)
		{
			std::cout << "no more particles can be spawned!\n";
			particleCap = i;
			break;
		}

		if (random)
		{
			partPos.x = ofRandom(cubeSize.x);
			partPos.y = ofRandom(cubeSize.y);
			partPos.z = ofRandom(cubeSize.z);
		}

		mPosition[mNumberOfParticles + i] = cubePos + partPos;
		mVelocity[mNumberOfParticles + i] = ofVec3f(0.f);

		if(random == false)
		{
			partPos.x += gap;

			if (partPos.x > cubeSize.x)
			{
				partPos.x = 0.f;
				partPos.z += gap;

				if (partPos.z > cubeSize.z)
				{
					partPos.z = 0.f;
					partPos.y += gap;

					if (partPos.y > cubeSize.y)
					{
						std::cout << "addCube: w00t? only " << i << " particles?\n";
						particleCap = i;
						break;
					}
				}
			}
		}
	}

	//if "particleCap" is still -1, all particles got spawned
	if (particleCap == -1)
		particleCap = particleAmount;

	// sync the particles to the corresponding buffers
	if (mMode == ComputeMode::COMPUTE_SHADER/* || mMode == ComputeMode::CUDA*/)
	{
		mComputeData.positionBuffer.updateData(sizeof(ofVec4f) * mNumberOfParticles, sizeof(ofVec4f) * particleCap, mPosition + mNumberOfParticles);
		mComputeData.velocityBuffer.updateData(sizeof(ofVec4f) * mNumberOfParticles, sizeof(ofVec4f) * particleCap, mVelocity + mNumberOfParticles);
	}
	else if (mMode == ComputeMode::OPENCL)
	{
		//the first write does not need to block, because the second write blocks and that implicitly flushes the whole commandQueue
		mOCLHelper.getCommandQueue().enqueueWriteBuffer(mOCLData.positionBuffer, CL_FALSE, sizeof(ofVec4f) * mNumberOfParticles, particleCap * sizeof(ofVec4f), mPosition + mNumberOfParticles);
		mOCLHelper.getCommandQueue().enqueueWriteBuffer(mOCLData.velocityBuffer, CL_TRUE, sizeof(ofVec4f) * mNumberOfParticles, particleCap * sizeof(ofVec4f), mVelocity + mNumberOfParticles);
	}
	else if (mMode == ComputeMode::CUDA)
	{
		//memcpy(mCUData.position + mNumberOfParticles, mPosition + mNumberOfParticles, sizeof(ofVec4f) * particleCap);
		//cudaDeviceSynchronize();

		//map the OpenGL buffers to CUDA device pointers
		cudaGraphicsMapResources(1, &mCUData.cuPos);
		cudaGraphicsMapResources(1, &mCUData.cuVel);

		size_t cudaPosSize, cudaVelSize;

		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&mCUData.position, &cudaPosSize, mCUData.cuPos));
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&mCUData.velocity, &cudaVelSize, mCUData.cuVel));

		checkCudaErrors(cudaMemcpy(mCUData.position + mNumberOfParticles, mPosition + mNumberOfParticles, sizeof(ofVec4f) * (particleCap), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(mCUData.velocity + mNumberOfParticles, mVelocity + mNumberOfParticles, sizeof(ofVec4f) * (particleCap), cudaMemcpyHostToDevice));
		//checkCudaErrors(cudaMemcpy(mCUData.position + mNumberOfParticles, mPosition + mNumberOfParticles, sizeof(ofVec4f) * particleCap, cudaMemcpyHostToDevice));
		//mComputeData.positionOutBuffer.copyTo(mComputeData.positionBuffer);

		//unmap all resources
		checkCudaErrors(cudaGraphicsUnmapResources(1, &mCUData.cuPos));
		checkCudaErrors(cudaGraphicsUnmapResources(1, &mCUData.cuVel));
	}

	mNumberOfParticles += particleCap;
}

void ParticleSystem::draw()
{
	mParticlesVBO.draw(GL_POINTS, 0, mNumberOfParticles);
}

ofVec3f ParticleSystem::getDimensions()
{
	return mDimension;
}

uint ParticleSystem::getNumberOfParticles()
{
	return mNumberOfParticles;
}

uint ParticleSystem::getCapacity()
{
	return mCapacity;
}

ParticleSystem::ComputeMode ParticleSystem::getMode()
{
	return mMode;
}

CUDAta& ParticleSystem::getCudata()
{
	return mCUData;
}

void ParticleSystem::measureNextUpdate()
{
	//std::cout << "Measuring next update!\n";
	mMeasureTime = true;
}

void ParticleSystem::update(float dt)
{
	// don't do anything when there are no particles present
	if (mNumberOfParticles == 0)
		return;

	uint64_t cycle;
	if (mMeasureTime)
		//mClock.start();
		cycle = __rdtsc();

	switch (mMode)
	{
		case ComputeMode::CPU:
			iUpdateCPU(dt);
			break;
		case ComputeMode::COMPUTE_SHADER:
			iUpdateCompute(dt);
			break;
		case ComputeMode::OPENCL:
			iUpdateOCL(dt);
			break;
		case ComputeMode::CUDA:
			iUpdateCUDA(dt);
			break;
	}

	if (mMeasureTime)
	{
		//double time = mClock.getDuration(mClock.stop());
		//std::cout << time << std::endl;
		cycle = __rdtsc() - cycle;
		std::cout << /*mNumberOfParticles << ";" <<*/ cycle << std::endl;
		mMeasureTime = false;
	}
}

void ParticleSystem::iUpdateCPU(float dt)
{
	float maxSpeed = 500.f;

	//optimization
	maxSpeed = 1.f / maxSpeed;

#pragma omp parallel for
	for (int i = 0; uint(i) < mNumberOfParticles; ++i)//warning: i can't be uint, because OMP needs an int (fix how?)
	{
		ofVec3f particlePosition = mPosition[i];
		ofVec3f particleVelocity = mVelocity[i];
		//ofVec3f particlePressure = iCalculatePressureVector(i);

		//gravity
		//particleVelocity += (mGravity + particlePressure) * dt;
		particleVelocity += (mGravity) * dt;
		//***g

		iApplyViscosity(i, dt, particleVelocity, particlePosition);

		//static collision
		for (int i = 0; i < 3; ++i)
		{
			if ((particlePosition[i] + particleVelocity[i] * dt > mDimension[i] && particleVelocity[i] > 0.f) // max boundary
				|| (particlePosition[i] + particleVelocity[i] * dt < 0.f && particleVelocity[i] < 0.f) // min boundary
				)
			{
				if (particlePosition[i] + particleVelocity[i] * dt < 0.f)
					particlePosition[i] = 0.f;
				else
					particlePosition[i] = mDimension[i];

				particleVelocity[i] *= -.3f;
			}
		}
		//*** sc

		//particleVelocity += dt * particleVelocity * -0.01f;//damping
		particlePosition += particleVelocity * dt;

		mVelocity[i] = particleVelocity;
		mPosition[i] = particlePosition;
	}

	mComputeData.positionBuffer.updateData(mNumberOfParticles * sizeof(ofVec4f), mPosition);
}


ofVec3f ParticleSystem::iCalculatePressureVector(size_t index)
{
	float interactionRadius = mSimData.interactionRadius;
	//float amplitude = 1.f;
	ofVec3f particlePosition = mPosition[index];

	ofVec3f pressureVec;
	for (uint i = 0; i < mNumberOfParticles; ++i)
	{
		if (index == i)
			continue;

		ofVec3f dirVec = particlePosition - mPosition[i];
		float dist = dirVec.length();

		if (dist > interactionRadius * 1.f)
			continue;

		float pressure = 1.f - (dist / interactionRadius);
		//float pressure = amplitude * expf(-dist / interactionRadius);
		//pressureVec += pressure * vectorMath::normalize(dirVec);
		pressureVec += pressure * dirVec.getNormalized();
	}
	return pressureVec;
}

void ParticleSystem::iApplyViscosity(size_t index, float dt, OUT ofVec3f& velocity, OUT ofVec3f& position)
{
	ofVec3f vel = velocity;
	ofVec3f pos = mPosition[index];
	float alpha = 1.f;
	float beta = 1.f;
	float rho = 0.0f;
	float rhoNear = 0.0f;

	for (int i = 0; i < mNumberOfParticles; i++)
	{
		if (index == i)
			continue;

		ofVec3f dirVec = pos - mPosition[i];
		float dist = dirVec.length();
		mPosition[i].w = dist;

		//if (dist > interactionRadius * 1.0f || dist < 0.01f)
		if (dist > mSimData.interactionRadius)
			continue;

		ofVec3f dirVecN = dirVec.normalized();
		float moveDir = (vel - mVelocity[i]).dot(dirVecN);
		float distRel = dist / mSimData.interactionRadius;

		// viscosity
		if (moveDir > 0)
		{
			ofVec3f impulse = (1.f - distRel) * (alpha * moveDir + beta * moveDir * moveDir) * dirVecN * dt;
			//vel -= impulse * 0.5f;//goes back to the caller-particle
			//mVelocity[i] += impulse * 0.5f;//changes neighbour velocity directly
		}
		// *** v

		// double relaxation
		float distRel2 = (1.f - distRel) * (1.f - distRel);
		rho += distRel2; // density, uses quadratic kernel
		rhoNear += distRel2 * (1.f - distRel);// near density, cubic kernel
		// *** dr
	}

	float pressure = (rho - mSimData.rho0) * mSimData.spring;
	float pressureNear = rhoNear * mSimData.springNear;

	ofVec3f displace;
	for (int i = 0; i < mNumberOfParticles; i++)
	{
		if (index == i)
			continue;

		float dist = mPosition[i].w;

		//if (dist > interactionRadius * 1.0f || dist < 0.01f)
		if (dist > mSimData.interactionRadius)
			continue;

		float distRel = dist / mSimData.interactionRadius;
		ofVec3f dirVecN = (pos - mPosition[i]).normalized();

		float pressureEffect = pressure * (1.f - distRel);
		float pressureNearEffect = pressureNear * (1.f - distRel) * (1.f - distRel);
		float df = (pressureEffect + pressureNearEffect) * dt * dt;
		ofVec3f d = dirVecN * df;
		//mPosition[i] += (d * 0.5f);
		displace -= d;
	}

	//pos += displace;
	vel += displace;

	position = pos;
	velocity = vel;

	//result[0].w = rho;
	//position.w = displace.length();
}

void ParticleSystem::iUpdateCompute(float dt)
{
	/*ofVec4f* tmpPositionFromGPU;
	tmpPositionFromGPU = mPositionBuffer.map<ofVec4f>(GL_READ_ONLY);
	mPositionBuffer.unmap();*///keep this snippet here for copy-pasta if something fails

	mComputeData.positionBuffer.bindBase(GL_SHADER_STORAGE_BUFFER, 0);
	mComputeData.positionOutBuffer.bindBase(GL_SHADER_STORAGE_BUFFER, 1);
	mComputeData.velocityBuffer.bindBase(GL_SHADER_STORAGE_BUFFER, 2);

	mComputeData.computeShader.begin();
	mComputeData.computeShader.setUniform1f("dt", dt);
	mComputeData.computeShader.setUniform1f("interactionRadius", mSimData.interactionRadius);
	mComputeData.computeShader.setUniform1f("rho0", mSimData.rho0);
	mComputeData.computeShader.setUniform1f("spring", mSimData.spring);
	mComputeData.computeShader.setUniform1f("springNear", mSimData.springNear);
	mComputeData.computeShader.setUniform3f("gravity", mGravity);
	mComputeData.computeShader.setUniform1i("numberOfParticles", mNumberOfParticles);
	mComputeData.computeShader.setUniform3f("mDimension", mDimension);
	mComputeData.computeShader.dispatchCompute(std::ceilf(float(mNumberOfParticles)/512), 1, 1);
	mComputeData.computeShader.end();

	mComputeData.positionOutBuffer.copyTo(mComputeData.positionBuffer);//TODO: swap instead of copy buffers

	//ofVec4f* tmpPositionFromGPU;
	//tmpPositionFromGPU = mComputeData.positionBuffer.map<ofVec4f>(GL_READ_ONLY);
	//for (uint i = 0; i < mNumberOfParticles; i++)
	//{
	//	if (isnan(tmpPositionFromGPU[i].x))
	//	{
	//		__debugbreak();
	//	}
	//}
	//mComputeData.positionBuffer.unmap();//*//keep this snippet here for copy-pasta if something fails
}

void ParticleSystem::iUpdateOCL(float dt)
{
	cl_int err;
	cl::CommandQueue queue = mOCLHelper.getCommandQueue();
	cl::Kernel kernel = mOCLHelper.getKernel();

	// set all the kernel arguments
	kernel.setArg(0, mOCLData.positionBuffer);
	kernel.setArg(1, mOCLData.positionOutBuffer);
	kernel.setArg(2, mOCLData.velocityBuffer);
	kernel.setArg(3, dt);
	kernel.setArg(4, mSimData.interactionRadius);
	kernel.setArg(5, ofVec4f(mGravity));
	kernel.setArg(6, ofVec4f(mDimension));
	kernel.setArg(7, mNumberOfParticles);

	cl::NDRange local;
	cl::NDRange global;
	cl::NDRange offset(0);

	if (mNumberOfParticles < mOCLData.maxWorkGroupSize)
	{
		local = cl::NDRange(mNumberOfParticles);
		global = cl::NDRange(mNumberOfParticles);
	}
	else
	{
		size_t f = std::ceilf(float(mNumberOfParticles) / mOCLData.maxWorkGroupSize);
		local = cl::NDRange(mOCLData.maxWorkGroupSize);
		global = cl::NDRange(mOCLData.maxWorkGroupSize * f);
		//global = cl::NDRange(mNumberOfParticles);
	}
	
	err = queue.enqueueNDRangeKernel(kernel, offset, global, local);
	oclHelper::handle_clerror(err, __LINE__);

	err = queue.enqueueReadBuffer(mOCLData.positionOutBuffer, CL_TRUE, 0, mNumberOfParticles * sizeof(ofVec4f), mPosition);
	oclHelper::handle_clerror(err, __LINE__);

	err = queue.enqueueCopyBuffer(mOCLData.positionOutBuffer, mOCLData.positionBuffer, 0, 0, mNumberOfParticles * sizeof(ofVec4f));
	oclHelper::handle_clerror(err, __LINE__);

	mComputeData.positionBuffer.updateData(mNumberOfParticles * sizeof(ofVec4f), mPosition);
}

void ParticleSystem::iUpdateCUDA(float dt)
{
	//convert some host variables to device types
	float3 cudaGravity = make_float3(mGravity.x, mGravity.y, mGravity.z);
	float3 cudaDimension = make_float3(mDimension.x, mDimension.y, mDimension.z);

	//map the OpenGL buffers to CUDA device pointers
	checkCudaErrors(cudaGraphicsMapResources(1, &mCUData.cuPos));
	checkCudaErrors(cudaGraphicsMapResources(1, &mCUData.cuPosOut));
	checkCudaErrors(cudaGraphicsMapResources(1, &mCUData.cuVel));

	size_t cudaPosSize, cudaPosOutSize, cudaVelSize;

	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&mCUData.position, &cudaPosSize, mCUData.cuPos));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&mCUData.positionOut, &cudaPosOutSize, mCUData.cuPosOut));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&mCUData.velocity, &cudaVelSize, mCUData.cuVel));

	//call the kernel
	cudaUpdate(mCUData.position, mCUData.positionOut, mCUData.velocity, dt, mSimData.interactionRadius, cudaGravity, cudaDimension, mNumberOfParticles);

	//unmap all resources
	checkCudaErrors(cudaGraphicsUnmapResources(1, &mCUData.cuPos));
	checkCudaErrors(cudaGraphicsUnmapResources(1, &mCUData.cuPosOut));
	checkCudaErrors(cudaGraphicsUnmapResources(1, &mCUData.cuVel));

	//sync the device data back to the host and write into the OpenGL buffer (for drawing)
	mComputeData.positionOutBuffer.copyTo(mComputeData.positionBuffer);
}

// debug function to count how many particles are outside the boundary
uint ParticleSystem::debug_testIfParticlesOutside()
{
	uint count = 0;
	for (uint i = 0; i < mNumberOfParticles; ++i)
	{
		ofVec3f particlePosition = mPosition[i];
		if (particlePosition.x > mDimension.x || particlePosition.x < 0.f
			|| particlePosition.y > mDimension.y || particlePosition.y < 0.f
			|| particlePosition.z > mDimension.z || particlePosition.z < 0.f)
		{
			count++;
			//__debugbreak();
		}
	}
	return count;
}
