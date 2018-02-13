#include "ParticleSystem.h"

ParticleSystem::ParticleSystem(uint maxParticles)
	:mGravity(0.f, -9.81f, 0.f),
	mCapacity(maxParticles),
	mNumberOfParticles(0),
	mMode(ComputeMode::CPU)
{
	//*** setup for CPU
	mPosition = new ofVec4f[maxParticles];
	mVelocity = new ofVec4f[maxParticles];
	mAvailableModes[ComputeMode::CPU] = true;
	//***

	//*** setup for Compute Shader
	mComputeData.positionOutBuffer.allocate(sizeof(ofVec4f) * maxParticles, mPosition, GL_DYNAMIC_DRAW);
	mComputeData.positionBuffer.allocate(sizeof(ofVec4f) * maxParticles, mPosition, GL_DYNAMIC_DRAW);
	mComputeData.velocityBuffer.allocate(sizeof(ofVec4f) * maxParticles, mVelocity, GL_DYNAMIC_DRAW);

	mParticlesVBO.setVertexBuffer(mComputeData.positionBuffer, 3, sizeof(ofVec4f));//use compute shader buffer to store the particle position on the GPU

	if (!mComputeData.computeShader.setupShaderFromFile(GL_COMPUTE_SHADER, "particles.compute"))
		mAvailableModes[ComputeMode::COMPUTE_SHADER] = false;
	else
		mAvailableModes[ComputeMode::COMPUTE_SHADER] = true;
	mComputeData.computeShader.linkProgram();
	//***

	//*** setup for OpenCL
	const char* sourceFile = "data/particles.cl";
	if (!mOCLHelper.setupOpenCLContext(1))
	{
		if (mOCLHelper.compileKernel(sourceFile))
		{
			std::cout << "ERROR: Unable to compile \"" << sourceFile << "\".\n";
			mAvailableModes[ComputeMode::OPENCL] = false;
		}
		else
		{
			cl::Context context = mOCLHelper.getCLContext();
			mOCLData.positionBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(ofVec4f) * maxParticles);
			mOCLData.positionOutBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(ofVec4f) * maxParticles);
			mOCLData.velocityBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(ofVec4f) * maxParticles);
			mAvailableModes[ComputeMode::OPENCL] = true;
		}
	}
	else
	{
		std::cout << "ERROR: Unable to create OpenCL context\n";
	}

	cl_int err = mOCLHelper.getDevice().getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &mOCLData.maxWorkGroupSize);
	mOCLData.maxWorkGroupSize /= 2;
	oclHelper::handle_clerror(err, __LINE__);
	//***

	//*** setup for CUDA
	// !TODO!
	const char* cmdArgs = "";//TODO: include CUDA cmdline arguments via settings file
	findCudaDevice(0, &cmdArgs);
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&mCUData.cuPos, mComputeData.positionBuffer.getId(), cudaGraphicsMapFlagsNone));
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&mCUData.cuVel, mComputeData.velocityBuffer.getId(), cudaGraphicsMapFlagsNone));
	//checkCudaErrors(cudaMallocManaged(&mCUData.position, sizeof(float4) * maxParticles));
	//checkCudaErrors(cudaMallocManaged(&mCUData.velocity, sizeof(float4) * maxParticles));
	mAvailableModes[ComputeMode::CUDA] = true;
	//***
}

ParticleSystem::~ParticleSystem()
{
	delete[] mPosition;
	delete[] mVelocity;
	//delete[] mPressure;

	cudaFree(mCUData.position);
	cudaFree(mCUData.velocity);
}

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
	//first sync all data back to RAM
	if (mMode == ComputeMode::COMPUTE_SHADER || m == ComputeMode::CUDA)
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
	/*else if (m == ComputeMode::CUDA)
	{
		cudaDeviceSynchronize();
		memcpy(mPosition, mCUData.position, sizeof(ofVec4f) * mNumberOfParticles);
	}*/

	//and then copy the data to the corresponding buffer for the new mode
	if (m == ComputeMode::COMPUTE_SHADER)
	{
		mComputeData.positionBuffer.updateData(sizeof(ofVec4f) * mNumberOfParticles, mPosition);
		mComputeData.velocityBuffer.updateData(sizeof(ofVec4f) * mNumberOfParticles, mVelocity);
	}
	else if (m == ComputeMode::OPENCL)
	{
		mOCLHelper.getCommandQueue().enqueueWriteBuffer(mOCLData.positionBuffer, CL_TRUE, 0, mNumberOfParticles * sizeof(ofVec4f), mPosition);
		mOCLHelper.getCommandQueue().enqueueWriteBuffer(mOCLData.velocityBuffer, CL_TRUE, 0, mNumberOfParticles * sizeof(ofVec4f), mVelocity);
	}
	/*else if (m == ComputeMode::CUDA)
	{
		memcpy(mCUData.position, mPosition, sizeof(ofVec4f) * mNumberOfParticles);
		cudaDeviceSynchronize();
	}*/
/*
	if (mMode == ComputeMode::COMPUTE_SHADER && m == ComputeMode::CPU)
	{
		ofVec4f* tmpPositionFromGPU = mComputeData.positionBuffer.map<ofVec4f>(GL_READ_ONLY);
		std::copy(tmpPositionFromGPU, tmpPositionFromGPU + mNumberOfParticles, mPosition);
		mComputeData.positionBuffer.unmap();
	}
	else if (mMode == ComputeMode::CPU && m == ComputeMode::COMPUTE_SHADER)
	{
		mComputeData.positionBuffer.updateData(sizeof(ofVec4f) * mNumberOfParticles, mPosition);
		mComputeData.velocityBuffer.updateData(sizeof(ofVec4f) * mNumberOfParticles, mVelocity);
	}*/

	mMode = m;
}

ParticleSystem::ComputeMode ParticleSystem::nextMode(ParticleSystem::ComputeMode current)
{
	int mode = static_cast<int>(current);
	mode++;
	if (mode >= static_cast<int>(ParticleSystem::ComputeMode::COMPUTEMODES_SIZE))
	{
		mode = 0;
	}

	if (!mAvailableModes[static_cast<ParticleSystem::ComputeMode>(mode)])
		mode = static_cast<int>(nextMode(static_cast<ParticleSystem::ComputeMode>(mode)));

	return static_cast<ParticleSystem::ComputeMode>(mode);
}

void ParticleSystem::setSmoothingWidth(float sw)
{
	mSimData.smoothingWidth = sw;
}

void ParticleSystem::addRandom(uint particleAmount)
{
	if (mCapacity < mNumberOfParticles + particleAmount)
	{
		std::cout << "no more particles can be spawned!\n";
		return;
	}

	ofSeedRandom();
	for (uint i = 0; i < particleAmount; i++)
	{
		mPosition[mNumberOfParticles + i].x = ofRandom(mDimension.x);
		mPosition[mNumberOfParticles + i].y = ofRandom(mDimension.y);
		mPosition[mNumberOfParticles + i].z = ofRandom(mDimension.z);
	}

	mNumberOfParticles += particleAmount;
}

void ParticleSystem::addDamBreak(uint particleAmount)
{
	ofVec3f damSize = mDimension;
	damSize.x *= 0.2f;
	addCube(ofVec3f(0) + 0.1f, damSize - 0.1f, particleAmount);
}

void ParticleSystem::addCube(ofVec3f cubePos, ofVec3f cubeSize, uint particleAmount)
{
	/*if (mCapacity < mNumberOfParticles + particleAmount)
	{
		std::cout << "no more particles can be spawned!\n";
		return;
	}*/

	/*if (mMode != ComputeMode::CPU && mNumberOfParticles > 0)
	{
		ofVec4f* tmpPositionFromGPU = mComputeData.positionBuffer.map<ofVec4f>(GL_READ_ONLY);
		std::copy(tmpPositionFromGPU, tmpPositionFromGPU + mNumberOfParticles, mPosition);
		mComputeData.positionBuffer.unmap();

		tmpPositionFromGPU = mComputeData.velocityBuffer.map<ofVec4f>(GL_READ_ONLY);
		std::copy(tmpPositionFromGPU, tmpPositionFromGPU + mNumberOfParticles, mVelocity);
		mComputeData.velocityBuffer.unmap();
	}*/

	float volume = cubeSize.x * cubeSize.y * cubeSize.z;
	//float spacePerParticle = volume / particleAmount;
	float cubedParticles = powf(particleAmount, 1.f / 3.f) * 3;
	float ratio;
	ratio = cubeSize.x / (cubeSize.x + cubeSize.y + cubeSize.z);
	float gap;
	gap = cubeSize.x / (cubedParticles * ratio);

	ofVec3f partPos(0.f);

	int particleCap = -1;
	for (uint i = 0; i < particleAmount; i++)
	{
		if (mNumberOfParticles + i >= mCapacity)
		{
			std::cout << "no more particles can be spawned!\n";
			particleCap = i;
			break;
		}

		mPosition[mNumberOfParticles + i] = cubePos + partPos;
		mVelocity[mNumberOfParticles + i] = ofVec3f(0.f);

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

	if (particleCap == -1)
		particleCap = particleAmount;

	if (mMode == ComputeMode::COMPUTE_SHADER)
	{
		mComputeData.positionBuffer.updateData(sizeof(ofVec4f) * mNumberOfParticles, sizeof(ofVec4f) * particleCap, mPosition + mNumberOfParticles);
		mComputeData.velocityBuffer.updateData(sizeof(ofVec4f) * mNumberOfParticles, sizeof(ofVec4f) * particleCap, mVelocity + mNumberOfParticles);
	}
	else if (mMode == ComputeMode::OPENCL)
	{
		mOCLHelper.getCommandQueue().enqueueWriteBuffer(mOCLData.positionBuffer, CL_TRUE, sizeof(ofVec4f) * mNumberOfParticles, particleCap * sizeof(ofVec4f), mPosition + mNumberOfParticles);
		mOCLHelper.getCommandQueue().enqueueWriteBuffer(mOCLData.velocityBuffer, CL_TRUE, sizeof(ofVec4f) * mNumberOfParticles, particleCap * sizeof(ofVec4f), mVelocity + mNumberOfParticles);
	}
	else if (mMode == ComputeMode::CUDA)
	{
		memcpy(mCUData.position + mNumberOfParticles, mPosition + mNumberOfParticles, sizeof(ofVec4f) * particleCap);
		cudaDeviceSynchronize();
	}

	mNumberOfParticles += particleCap;
}

void ParticleSystem::addDrop()
{
	if (mCapacity < mNumberOfParticles + 20)
		return;
	
	mNumberOfParticles += 20;
}

void ParticleSystem::draw()
{
	/*if(mShaderStorageSwap)
		mParticlesVBO.setVertexBuffer(mPositionBuffer, 3, sizeof(ofVec4f), 0);
	else
		mParticlesVBO.setVertexBuffer(mPositionOutBuffer, 3, sizeof(ofVec4f), 0);*/
	mParticlesVBO.draw(GL_POINTS, 0, mNumberOfParticles);
}

ofVec4f * ParticleSystem::getPositionPtr()
{
	return mPosition;
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

void ParticleSystem::update(float dt)
{
	if (mNumberOfParticles == 0)
		return;

	mGravityRotated = mGravity;
	if (!mRotation.zeroRotation())
	{
		ofVec3f axis;
		float angle;
		mRotation.getRotate(angle, axis);
		mGravityRotated = mGravity.rotate(angle, axis);
	}

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
		ofVec3f particlePressure = iCalculatePressureVector(i);

		//float r = m_randoms[i];
		//m_rng.seed(i * 815, 1337, 420);

		//std::cout << vectorMath::angleD(vectorMath::normalize(newVel) - vectorMath::normalize(m_velocity[i])) << std::endl;
		//std::cout << vectorMath::radToDeg(atan2f(newVel.y - m_velocity[i].y, newVel.x - m_velocity[i].x)) << std::endl;

		//gravity
		/*if	  (ofRectangle(-0.1f, -0.1f, mDimension.x + 0.1f, mDimension.y + 0.1f).inside(particlePosition.x, particlePosition.y)
		&& ofRectangle(-0.1f, -0.1f, mDimension.x + 0.1f, mDimension.z + 0.1f).inside(particlePosition.x, particlePosition.z)
		&& ofRectangle(-0.1f, -0.1f, mDimension.y + 0.1f, mDimension.z + 0.1f).inside(particlePosition.y, particlePosition.z))
		*/
		if (particlePosition.x <= mDimension.x || particlePosition.x >= 0.f
			|| particlePosition.y <= mDimension.y || particlePosition.y >= 0.f
			|| particlePosition.z <= mDimension.z || particlePosition.z >= 0.f)
			particleVelocity += (mGravityRotated + particlePressure) * dt;
		//***g

		//static collision
		for (int i = 0; i < 3; ++i)
		{
			if ((particlePosition[i] > mDimension[i] && particleVelocity[i] > 0.f) || (particlePosition[i] < 0.f && particleVelocity[i] < 0.f))
			{
				particleVelocity[i] *= -.3f;
			}
		}
		//*** sc

		//particleVelocity += dt * particleVelocity * -0.01f;//damping
		particlePosition += particleVelocity * dt;

		mVelocity[i] = particleVelocity;
		mPosition[i] = particlePosition;

		//m_vertices[i].position = particlePosition;
	}

	mComputeData.positionBuffer.updateData(mNumberOfParticles * sizeof(ofVec4f), mPosition);
}


ofVec3f ParticleSystem::iCalculatePressureVector(size_t index)
{
	float smoothingWidth = mSimData.smoothingWidth;
	//float amplitude = 1.f;
	ofVec3f particlePosition = mPosition[index];

	ofVec3f pressureVec;
	for (uint i = 0; i < mNumberOfParticles; ++i)
	{
		if (index == i)
			continue;

		ofVec3f dirVec = particlePosition - mPosition[i];
		float dist = dirVec.length();

		if (dist > smoothingWidth * 1.f)
			continue;

		float pressure = 1.f - (dist / smoothingWidth);
		//float pressure = amplitude * expf(-dist / smoothingWidth);
		//pressureVec += pressure * vectorMath::normalize(dirVec);
		pressureVec += pressure * dirVec.getNormalized();
	}
	return pressureVec;
}

void ParticleSystem::iUpdateCompute(float dt)
{
	/*ofVec4f* tmpPositionFromGPU;
	tmpPositionFromGPU = mPositionBuffer.map<ofVec4f>(GL_READ_ONLY);
	mPositionBuffer.unmap();*/

	/*tmpPositionFromGPU = mVelocityBuffer.map<ofVec4f>(GL_READ_ONLY);
	mVelocityBuffer.unmap();*/

	//mPositionBuffer.updateData(sizeof(ofVec3f) * mNumberOfParticles, mPosition);

	mComputeData.positionBuffer.bindBase(GL_SHADER_STORAGE_BUFFER, 0);
	mComputeData.positionOutBuffer.bindBase(GL_SHADER_STORAGE_BUFFER, 1);
	mComputeData.velocityBuffer.bindBase(GL_SHADER_STORAGE_BUFFER, 2);

	mComputeData.computeShader.begin();
	mComputeData.computeShader.setUniform1f("dt", dt);
	mComputeData.computeShader.setUniform1f("smoothingWidth", mSimData.smoothingWidth);
	mComputeData.computeShader.setUniform3f("gravity", mGravityRotated);
	mComputeData.computeShader.setUniform1i("numberOfParticles", mNumberOfParticles);
	mComputeData.computeShader.setUniform3f("mDimension", mDimension);
	if (std::ceilf(float(mNumberOfParticles) / 512) < 1.f)
		__debugbreak();
	mComputeData.computeShader.dispatchCompute(std::ceilf(float(mNumberOfParticles)/512), 1, 1);
	mComputeData.computeShader.end();

	mComputeData.positionOutBuffer.copyTo(mComputeData.positionBuffer);//TODO: swap instead of copy buffers

	/*tmpPositionFromGPU = mPositionOutBuffer.map<ofVec4f>(GL_READ_ONLY);
	mPositionOutBuffer.unmap();//*/

	/*tmpPositionFromGPU = mVelocityBuffer.map<ofVec4f>(GL_READ_ONLY);
	mVelocityBuffer.unmap();//*/
}

void ParticleSystem::iUpdateOCL(float dt)
{
	cl_int err;
	cl::CommandQueue queue = mOCLHelper.getCommandQueue();
	cl::Kernel kernel = mOCLHelper.getKernel();

	kernel.setArg(0, mOCLData.positionBuffer);
	kernel.setArg(1, mOCLData.positionOutBuffer);
	kernel.setArg(2, mOCLData.velocityBuffer);
	kernel.setArg(3, dt);
	kernel.setArg(4, mSimData.smoothingWidth);
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
		size_t someDiff = std::ceilf(float(mNumberOfParticles) / mOCLData.maxWorkGroupSize);
		local = cl::NDRange(mOCLData.maxWorkGroupSize);
		global = cl::NDRange(mOCLData.maxWorkGroupSize * someDiff);
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
	float4 cudaGravity = make_float4(mGravity.x, mGravity.y, mGravity.z, 0);
	float4 cudaDimension = make_float4(mDimension.x, mDimension.y, mDimension.z, 0);

	//map the OpenGL buffers to CUDA device pointers
	cudaGraphicsMapResources(1, &mCUData.cuPos);
	cudaGraphicsMapResources(1, &mCUData.cuVel);

	size_t cudaPosSize, cudaVelSize;

	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&mCUData.position, &cudaPosSize, mCUData.cuPos));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&mCUData.velocity, &cudaVelSize, mCUData.cuVel));

	//call the kernel
	//cudaUpdate(mCUData.position, mCUData.velocity, dt, mSimData.smoothingWidth, cudaGravity, cudaDimension, mNumberOfParticles);
	cudaUpdate(mCUData.position, mCUData.velocity, dt, mSimData.smoothingWidth, cudaGravity, cudaDimension, mNumberOfParticles);

	//sync the device data back to the host and write into the OpenGL buffer (for drawing)
	//cudaDeviceSynchronize();
	//mComputeData.positionBuffer.updateData(mNumberOfParticles * sizeof(ofVec4f), mCUData.position);

	//unmap all resources
	cudaGraphicsUnmapResources(1, &mCUData.cuPos);
	cudaGraphicsUnmapResources(1, &mCUData.cuVel);
}

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
