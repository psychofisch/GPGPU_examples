#include "ParticleSystem.h"

ParticleSystem::ParticleSystem(uint maxParticles)
	:mCapacity(maxParticles),
	mNumberOfParticles(0),
	mMode(ComputeMode::CPU)
{
	//*** general setup
	// these buffers are required, even when CPU mode is not enabled
	// they are used as cache between different GPU modes
	mParticlePosition = new ofVec4f[maxParticles];
	mParticleVelocity = new ofVec4f[maxParticles];

	for (int i = 0; i < ComputeMode::COMPUTEMODES_SIZE; ++i)
		mAvailableModes[i] = false;

	mGravity = Particle::gravity;
	//***

	//*** general GL setup
	mParticlesBuffer.allocate(sizeof(ofVec4f) * mCapacity, mParticlePosition, GL_DYNAMIC_DRAW);
	//mParticlesBuffer.allocate(sizeof(ofVec4f) * mCapacity, GL_DYNAMIC_DRAW_ARB);
	//mParticlesBuffer.updateData(sizeof(ofVec4f) * mCapacity, mParticlePosition);
	HANDLE_GL_ERROR();

	mParticlesVBO.setVertexBuffer(mParticlesBuffer, 3, sizeof(ofVec4f));

	ofSpherePrimitive sphere;
	sphere.set(1.f, 3, ofPrimitiveMode::OF_PRIMITIVE_TRIANGLES);
	mParticleModel = sphere.getMesh();
	mParticleModel.enableColors();
	mParticleModel.enableNormals();
	mParticleModel.enableTextures();

	mParticleTmp.set(0.01f, 3);

	mMeasureTime = false;
}

ParticleSystem::~ParticleSystem()
{
	// free the CPU buffes
	delete[] mParticlePosition;
	delete[] mParticleVelocity;

	// free the CUDA buffers
	if (mAvailableModes[ComputeMode::CUDA])
	{
		CUDAERRORS(cudaFree(mCUData.position));
		CUDAERRORS(cudaFree(mCUData.velocity));
		CUDAERRORS(cudaFree(mCUData.positionOut));
	}

	// GL and OpenCL buffers clear themselves at destruction

	//Thrust
	if (mAvailableModes[ComputeMode::THRUST])
	{
		delete mThrustData;
	}
}

void ParticleSystem::setupAll(ofxXmlSettings & settings)
{
	setupCPU(settings);
	setupCompute(settings);
	setupCUDA(settings);
	setupOCL(settings);
	setupThrust(settings);

	HANDLE_GL_ERROR();
}

void ParticleSystem::setupCPU(ofxXmlSettings & settings)
{
	if (settings.getValue("CPU::ENABLED", false) == false)
		return;

	mThreshold = settings.getValue("CPU:THRESHOLD", 1000);

	mAvailableModes[ComputeMode::CPU] = settings.getValue("CPU:ENABLED", true);

	HANDLE_GL_ERROR();
}

void ParticleSystem::setupCompute(ofxXmlSettings & settings)
{
	if (settings.getValue("COMPUTE::ENABLED", false) == false)
		return;

	// compile the compute code
	if (mComputeData.computeShader.setupShaderFromFile(GL_COMPUTE_SHADER, settings.getValue("COMPUTE:SOURCE", "particles.compute"))
		&& mComputeData.computeShader.linkProgram())
	{
		//allocate buffer memory
		mComputeData.positionOutBuffer.allocate(sizeof(ofVec4f) * mCapacity, mParticlePosition, GL_DYNAMIC_DRAW);
		mComputeData.positionBuffer.allocate(sizeof(ofVec4f) * mCapacity, mParticlePosition, GL_DYNAMIC_DRAW);
		mComputeData.velocityBuffer.allocate(sizeof(ofVec4f) * mCapacity, mParticleVelocity, GL_DYNAMIC_DRAW);

		mAvailableModes[ComputeMode::COMPUTE_SHADER] = settings.getValue("COMPUTE:ENABLED", true);
	}
	else
	{
		mAvailableModes[ComputeMode::COMPUTE_SHADER] = false;
	}

	HANDLE_GL_ERROR();
}

void ParticleSystem::setupCUDA(ofxXmlSettings & settings)
{
	if (settings.getValue("CUDA::ENABLED", false) == false)
		return;

	//load CUDA command line arguments from settings file
	const int cmdArgc = settings.getValue("CUDA:ARGC", 0);
	const char* cmdArgs = settings.getValue("CUDA:ARGV", "").c_str();
	
	// find a CUDA device
	findCudaDevice(cmdArgc, &cmdArgs);

	// allocate memory
	// note: do not use the CUDAERRORS macro here, because the case when these allocations fail in release mode has to be handled
	checkCudaErrors(cudaMalloc(&mCUData.position, sizeof(ofVec4f) * mCapacity));
	checkCudaErrors(cudaMalloc(&mCUData.velocity, sizeof(ofVec4f) * mCapacity));
	checkCudaErrors(cudaMalloc(&mCUData.positionOut, sizeof(ofVec4f) * mCapacity));
	
	// "checkCudaErrors" will quit the program in case of a problem, so it is safe to assume that if the program reached this point CUDA will work
	mAvailableModes[ComputeMode::CUDA] = settings.getValue("CUDA:ENABLED", true);

	HANDLE_GL_ERROR();
}

void ParticleSystem::setupOCL(ofxXmlSettings & settings)
{
	if (settings.getValue("OCL::ENABLED", false) == false)
		return;

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

	HANDLE_GL_ERROR();
}

void ParticleSystem::setupThrust(ofxXmlSettings & settings)
{
	if (settings.getValue("THRUST::ENABLED", false) == false)
		return;

	/*mThrustData.position = thrust::device_malloc<float4>(mCapacity);
	mThrustData.positionOut = thrust::device_malloc<float4>(mCapacity);
	mThrustData.velocity = thrust::device_malloc<float4>(mCapacity);*/
	
	mThrustData = ThrustHelper::setup(mNumberOfParticles);

	mAvailableModes[ComputeMode::THRUST] = true;

	HANDLE_GL_ERROR();
}

void ParticleSystem::createParticleShader(std::string vert, std::string frag)
{
	mParticleShader.load(vert, frag);

	HANDLE_GL_ERROR();
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
	// sync velocity data back from the GPU to RAM (particle positions get synced back every frame)
	if (mMode == ComputeMode::COMPUTE_SHADER)
	{
		/*ofVec4f* tmpPtrFromGPU = mComputeData.positionBuffer.map<ofVec4f>(GL_READ_ONLY);
		std::copy(tmpPtrFromGPU, tmpPtrFromGPU + mNumberOfParticles, mParticlePosition);
		mComputeData.positionBuffer.unmap();*/

		ofVec4f* tmpPtrFromGPU = mComputeData.velocityBuffer.map<ofVec4f>(GL_READ_ONLY);
		std::copy(tmpPtrFromGPU, tmpPtrFromGPU + mNumberOfParticles, mParticleVelocity);
		mComputeData.velocityBuffer.unmap();
	}
	else if (mMode == ComputeMode::OPENCL)
	{
		//mOCLHelper.getCommandQueue().enqueueReadBuffer(mOCLData.positionOutBuffer, CL_TRUE, 0, mNumberOfParticles * sizeof(ofVec4f), mParticlePosition);
		mOCLHelper.getCommandQueue().enqueueReadBuffer(mOCLData.velocityBuffer, CL_TRUE, 0, mNumberOfParticles * sizeof(ofVec4f), mParticleVelocity);
	}
	else if (m == ComputeMode::CUDA)// keep this, just in case
	{
		CUDAERRORS(cudaMemcpy(mParticleVelocity, mCUData.velocity, sizeof(ofVec4f) * mNumberOfParticles, cudaMemcpyDeviceToHost));
		//memcpy(mParticlePosition, mCUData.position, sizeof(ofVec4f) * mNumberOfParticles);
	}

	// copy the data to the corresponding buffer for the new mode
	if (m == ComputeMode::COMPUTE_SHADER)
	{
		mComputeData.positionBuffer.updateData(sizeof(ofVec4f) * mNumberOfParticles, mParticlePosition);
		mComputeData.velocityBuffer.updateData(sizeof(ofVec4f) * mNumberOfParticles, mParticleVelocity);

		//ofVec4f* positionsFromGPU = mComputeData.positionBuffer.map<ofVec4f>(GL_READ_ONLY);//TODO: use mapRange
		//mComputeData.positionBuffer.unmap();
	}
	else if (m == ComputeMode::OPENCL)
	{
		mOCLHelper.getCommandQueue().enqueueWriteBuffer(mOCLData.positionBuffer, CL_FALSE, 0, mNumberOfParticles * sizeof(ofVec4f), mParticlePosition);
		mOCLHelper.getCommandQueue().enqueueWriteBuffer(mOCLData.velocityBuffer, CL_TRUE, 0, mNumberOfParticles * sizeof(ofVec4f), mParticleVelocity);
	}
	else if (m == ComputeMode::CUDA)
	{
		CUDAERRORS(cudaMemcpy(mCUData.position, mParticlePosition, sizeof(ofVec4f) * mNumberOfParticles, cudaMemcpyHostToDevice));
		CUDAERRORS(cudaMemcpy(mCUData.velocity, mParticleVelocity, sizeof(ofVec4f) * mNumberOfParticles, cudaMemcpyHostToDevice));
	}

	// set mode
	mMode = m;
}

void ParticleSystem::setPosition(ofVec3f p)
{
	mPosition = p;
}

void ParticleSystem::addPosition(ofVec3f p)
{
	setPosition(mPosition + p);
}

void ParticleSystem::setStaticCollision(std::vector<MinMaxData>& collision)
{
	mStaticCollision = collision;
}

void ParticleSystem::setGravity(ofVec3f g)
{
	mGravity = g;
}

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

void ParticleSystem::addDamBreak(uint particleAmount)
{
	ofVec3f damSize = mDimension;
	damSize.x *= 0.2f;
	addCube(ofVec3f(0) + 0.1f, damSize - 0.1f, particleAmount);
}

void ParticleSystem::addCube(ofVec3f cubePos, ofVec3f cubeSize, uint particleAmount, bool random)
{
	//some calculations to distribute the particles evenly
	float cubedParticles = powf(particleAmount, 1.f / 3.f) * 3;
	float ratio;
	ratio = cubeSize.x / (cubeSize.x + cubeSize.y + cubeSize.z);
	float gap;
	gap = cubeSize.x / (cubedParticles * ratio);

	ofVec3f partPos;
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
			partPos.x = mPosition.x + ofRandom(cubeSize.x);
			partPos.y = mPosition.y + ofRandom(cubeSize.y);
			partPos.z = mPosition.z + ofRandom(cubeSize.z);
		}

		mParticlePosition[mNumberOfParticles + i] = cubePos + partPos + mPosition;
		mParticleVelocity[mNumberOfParticles + i] = ofVec3f(0.f);

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
		mComputeData.positionBuffer.updateData(sizeof(ofVec4f) * mNumberOfParticles, sizeof(ofVec4f) * particleCap, mParticlePosition + mNumberOfParticles);
		mComputeData.velocityBuffer.updateData(sizeof(ofVec4f) * mNumberOfParticles, sizeof(ofVec4f) * particleCap, mParticleVelocity + mNumberOfParticles);
	}
	else if (mMode == ComputeMode::OPENCL)
	{
		//the first write does not need to block, because the second write blocks and that implicitly flushes the whole commandQueue
		mOCLHelper.getCommandQueue().enqueueWriteBuffer(mOCLData.positionBuffer, CL_FALSE, sizeof(ofVec4f) * mNumberOfParticles, particleCap * sizeof(ofVec4f), mParticlePosition + mNumberOfParticles);
		mOCLHelper.getCommandQueue().enqueueWriteBuffer(mOCLData.velocityBuffer, CL_TRUE, sizeof(ofVec4f) * mNumberOfParticles, particleCap * sizeof(ofVec4f), mParticleVelocity + mNumberOfParticles);
	}
	else if (mMode == ComputeMode::CUDA)
	{
		CUDAERRORS(cudaMemcpy(mCUData.position + mNumberOfParticles, mParticlePosition + mNumberOfParticles, sizeof(ofVec4f) * (particleCap), cudaMemcpyHostToDevice));
		CUDAERRORS(cudaMemcpy(mCUData.velocity + mNumberOfParticles, mParticleVelocity + mNumberOfParticles, sizeof(ofVec4f) * (particleCap), cudaMemcpyHostToDevice));
	}

	mNumberOfParticles += particleCap;

	// copy the particles to the GL buffer for drawing
	mParticlesBuffer.updateData(sizeof(ofVec4f) * mNumberOfParticles, mParticlePosition);

	HANDLE_GL_ERROR();
}

void ParticleSystem::draw()
{
	if (mNumberOfParticles == 0)
		return;
	
	// *** DESIRED METHOD but does not render correctly in Compute Shader and OpenCL mode
	//glEnable(GL_CULL_FACE);
	//glCullFace(GL_BACK);

	mParticleShader.begin();

	// create to scale the particles in the shader
	ofMatrix4x4 identity;
	identity.makeIdentityMatrix();
	identity.scale(ofVec3f(mSimData.interactionRadius * 0.1f));

	// bind the buffer positions
	//mParticlesBuffer.bindBase(GL_SHADER_STORAGE_BUFFER, 4);
	mParticlesVBO.getVertexBuffer().bindBase(GL_SHADER_STORAGE_BUFFER, 4);

	// set uniforms
	//mParticleShader.setUniform3f("systemPos", mPosition);
	mParticleShader.setUniform1i("mode", 1);
	mParticleShader.setUniformMatrix4f("scale", identity);
	mParticleShader.setUniform1i("particles", mNumberOfParticles);

	// draw particles
	mParticleModel.drawInstanced(OF_MESH_FILL, mNumberOfParticles);
	//mParticleModel.drawInstanced(OF_MESH_POINTS, mNumberOfParticles);

	// unbind and clean up
	//mParticlesBuffer.unbindBase(GL_SHADER_STORAGE_BUFFER, 4);
	mParticlesVBO.getVertexBuffer().unbindBase(GL_SHADER_STORAGE_BUFFER, 4);

	mParticleShader.end();

	//glDisable(GL_CULL_FACE);
	// ***

	HANDLE_GL_ERROR();
}

ofVec3f ParticleSystem::getDimensions() const
{
	return mDimension;
}

uint ParticleSystem::getNumberOfParticles() const
{
	return mNumberOfParticles;
}

uint ParticleSystem::getCapacity() const
{
	return mCapacity;
}

ParticleSystem::ComputeMode ParticleSystem::getMode() const
{
	return mMode;
}

Particle::CUDAta& ParticleSystem::getCudata()
{
	return mCUData;
}

ofVec3f ParticleSystem::getPosition() const
{
	return mPosition;
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

	// each update computes the new particle positions and stores them into mParticlePosition (on the CPU)
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
		case ComputeMode::THRUST:
			iUpdateThrust(dt);
			break;
	}

	// copy CPU data to the GL buffer for drawing
	mParticlesBuffer.updateData(sizeof(ofVec4f) * mNumberOfParticles, mParticlePosition);

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
	float fluidDamp = 0.f;

	//optimization
	maxSpeed = 1.f / maxSpeed;
	ofVec3f boundingBox = mDimension;

#pragma omp parallel for
	for (int i = 0; uint(i) < mNumberOfParticles; ++i)//warning: i can't be uint, because OMP needs an int (fix how?)
	{
		ofVec3f particlePosition = mParticlePosition[i];
		ofVec3f particleVelocity = mParticleVelocity[i];
		
		// fluid simulation
		ofVec3f particlePressure = iCalculatePressureVector(i, particlePosition, particleVelocity);
		// *** fs

		// remove the system position offset
		particlePosition -= mPosition;

		// gravity
		particleVelocity += (mGravity + particlePressure) * dt;
		//particleVelocity += (mGravity) * dt;
		// ***g

		ofVec3f newpos = (particlePosition) + particleVelocity * dt;

		// bounding box collision
		for (int i = 0; i < 3; ++i)
		{
			if ((newpos[i] > boundingBox[i] && particleVelocity[i] > 0.f) // max boundary
				|| (newpos[i] < 0.f && particleVelocity[i] < 0.f) // min boundary
				)
			{
				if (newpos[i] < 0.f)
					particlePosition[i] = 0.f;
				else
					particlePosition[i] = boundingBox[i];

				particleVelocity[i] *= -fluidDamp;
			}
		}
		//*** bbc

		// static collision
		int collisionCnt = 3; //support 3 collisions at once
		for (size_t i = 0; i < mStaticCollision.size() && collisionCnt > 0; i++)
		{
			const ofVec3f currentMin = mStaticCollision[i].min;
			const ofVec3f currentMax = mStaticCollision[i].max;
			//ofVec3f otherMin, otherMax;
			int result = -1;

			if (((newpos.x < currentMax.x && newpos.x > currentMin.x))
				&&
				((newpos.z < currentMax.z && newpos.z > currentMin.z))
				&&
				((newpos.y < currentMax.y && newpos.y > currentMin.y))
				)
			{
				ofVec3f aabbCenter = currentMin + ((currentMax - currentMin)*0.5f);
				ofVec3f whichSide = (aabbCenter - newpos).normalized();
				int closest = 666;
				float current = -66.6;
				for (int i = 0; i < 6; ++i)
				{
					float angle = whichSide.dot(Particle::directions[i]);
					if (angle > current)
					{
						current = angle;
						closest = i;
					}
				}

				if (closest == 666)
				{
					std::cout << "W00T!?\n";
					__debugbreak();
				}

				//ofVec3f reflection;
				ofVec3f n = Particle::directions[closest];

				// source -> https://math.stackexchange.com/questions/13261/how-to-get-a-reflection-vector#13266
				//particleVelocity = (particleVelocity - (2 * particleVelocity.dot(n) * n)) + (n * 0.1f);
				particleVelocity = n;
				particleVelocity *= fluidDamp;
				
				collisionCnt = 0;
				//result = j;
				//break;// OPT: do not delete this (30% performance loss)
			}
		}
		// *** sc

		//particleVelocity += dt * particleVelocity * -0.01f;//damping
		particlePosition += particleVelocity * dt;

		mParticleVelocity[i] = particleVelocity;
		mParticlePosition[i] = particlePosition + mPosition; // add the system position offset
	}
}

ofVec3f ParticleSystem::iCalculatePressureVector(size_t index, ofVec4f pos, ofVec4f vel)
{
	float interactionRadius = mSimData.interactionRadius;
	interactionRadius *= interactionRadius;
	//float amplitude = 1.f;
	ofVec3f particlePosition = pos;
	float density = 0.f;
	float viscosity = mSimData.viscosity;
	float restPressure = mSimData.restPressure;
	float pressureMultiplier = mSimData.pressureMultiplier;

	ofVec3f pressureVec, viscosityVec;
	for (uint i = 0; i < mNumberOfParticles; ++i)
	{
		if (index == i)
			continue;

		ofVec3f dirVec = particlePosition - mParticlePosition[i];
		//float dist = dirVec.length();
		float dist = dirVec.lengthSquared();

		if (dist > interactionRadius)
			continue;

		ofVec3f dirVecN = dirVec.getNormalized();
		float moveDir = (vel - mParticleVelocity[i]).dot(dirVecN);
		float distRel = 1.f - (sqrtf(dist / interactionRadius));

		// viscosity
		if (true || moveDir > 0)
		{
			//ofVec3f impulse = (1.f - distRel) * (mSimData.viscosity * moveDir + mSimData.restPressure * moveDir * moveDir) * dirVecN;
			ofVec3f impulse = distRel * (viscosity * moveDir) * dirVecN;
			viscosityVec -= impulse * 0.5f;//goes back to the caller-particle
		}
		// *** v

		// pressure 
		float sqx = distRel * distRel;
		//float pressure = 1.f - pressureMultiplier * (sqx * oneminusx - sqx);
		float pressure = sqx * pressureMultiplier;
		// *** p

		// viscosity
		//float viscosity = mSimData.restPressure * (sqx);
		// *** s

		density += 1.f;

		//float pressure = 1.f - (dist / interactionRadius);
		//float pressure = amplitude * expf(-dist / interactionRadius);
		//pressureVec += pressure * vectorMath::normalize(dirVec);
		pressureVec += (pressure - restPressure) * dirVecN;
	}

	//pressureVec = pressureVec.length() * ((density / mSimData.restPressure) - 0.5f) * pressureVec.getNormalized();

	return pressureVec + viscosityVec;
}

void ParticleSystem::iUpdateCompute(float dt)
{
	/*ofVec4f* tmpPositionFromGPU;
	tmpPositionFromGPU = mPositionBuffer.map<ofVec4f>(GL_READ_ONLY);
	mPositionBuffer.unmap();*///keep this snippet here for copy-pasta if something fails

	// bind all required buffers and set uniforms
	mComputeData.positionBuffer.bindBase(GL_SHADER_STORAGE_BUFFER, 0);
	mComputeData.positionOutBuffer.bindBase(GL_SHADER_STORAGE_BUFFER, 1);
	mComputeData.velocityBuffer.bindBase(GL_SHADER_STORAGE_BUFFER, 2);

	mComputeData.computeShader.begin();
	mComputeData.computeShader.setUniform1f("dt", dt);
	mComputeData.computeShader.setUniform1f("interactionRadius", mSimData.interactionRadius);
	mComputeData.computeShader.setUniform1f("rho0", mSimData.pressureMultiplier);
	mComputeData.computeShader.setUniform1f("spring", mSimData.viscosity);
	mComputeData.computeShader.setUniform1f("springNear", mSimData.restPressure);
	mComputeData.computeShader.setUniform3f("gravity", mGravity);
	mComputeData.computeShader.setUniform3f("position", mPosition);
	mComputeData.computeShader.setUniform1i("numberOfParticles", mNumberOfParticles);//TODO: conversion from uint to int
	mComputeData.computeShader.setUniform3f("mDimension", mDimension);

	// call the kernel
	// local size: hard-coded "512", because it is also hard-coded in the kernel source code
	mComputeData.computeShader.dispatchCompute(std::ceilf(float(mNumberOfParticles)/512), 1, 1);
	mComputeData.computeShader.end();//forces the program to wait until the calculation is finished 

	// copy the new positions to the position Buffer
	mComputeData.positionOutBuffer.copyTo(mComputeData.positionBuffer);//TODO: swap instead of copy buffers

	// sync the result back to the CPU
	ofVec4f* positionsFromGPU = mComputeData.positionBuffer.map<ofVec4f>(GL_READ_ONLY);//TODO: use mapRange
	std::copy(positionsFromGPU, positionsFromGPU + mNumberOfParticles, mParticlePosition);
	mComputeData.positionBuffer.unmap();//*//keep this snippet here for copy-pasta if something fails

	mComputeData.positionBuffer.unbindBase(GL_SHADER_STORAGE_BUFFER, 0);
	mComputeData.positionOutBuffer.unbindBase(GL_SHADER_STORAGE_BUFFER, 1);
	mComputeData.velocityBuffer.unbindBase(GL_SHADER_STORAGE_BUFFER, 2);

	//ofVec4f* tmpPositionFromGPU;
	//tmpPositionFromGPU = mComputeData.velocityBuffer.map<ofVec4f>(GL_READ_ONLY);
	//int cnt = 0;
	//for (uint i = 0; i < mNumberOfParticles; i++)
	//{
	//	//if (isnan(tmpPositionFromGPU[i].x))
	//	if (tmpPositionFromGPU[i].w == 666.0f)
	//	{
	//		//__debugbreak();
	//		cnt++;
	//	}
	//}
	//if (cnt > 0)
	//	__debugbreak();
	//mComputeData.velocityBuffer.unmap();//*//keep this snippet here for copy-pasta if something fails
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
	kernel.setArg(4, ofVec4f(mGravity));
	kernel.setArg(5, ofVec4f(mPosition));
	kernel.setArg(6, ofVec4f(mDimension));
	kernel.setArg(7, mNumberOfParticles);
	kernel.setArg(8, mSimData);

	// calculate the global and local size
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
	
	// call the kernel
	err = queue.enqueueNDRangeKernel(kernel, offset, global, local);
	oclHelper::handle_clerror(err, __LINE__);

	// copy the result back to the CPU
	err = queue.enqueueReadBuffer(mOCLData.positionOutBuffer, CL_TRUE, 0, mNumberOfParticles * sizeof(ofVec4f), mParticlePosition);
	oclHelper::handle_clerror(err, __LINE__);

	// copy the result to the GPU position buffer
	err = queue.enqueueCopyBuffer(mOCLData.positionOutBuffer, mOCLData.positionBuffer, 0, 0, mNumberOfParticles * sizeof(ofVec4f));
	oclHelper::handle_clerror(err, __LINE__);

	// wait until the queue is finished
	err = queue.finish();
	oclHelper::handle_clerror(err, __LINE__);
}

void ParticleSystem::iUpdateCUDA(float dt)
{
	// convert some host variables to device types
	float3 cudaGravity = make_float3(mGravity.x, mGravity.y, mGravity.z);
	float3 cudaDimension = make_float3(mDimension.x, mDimension.y, mDimension.z);
	float3 cudaPosition = make_float3(mPosition.x, mPosition.y, mPosition.z);

	// call the kernel
	cudaParticleUpdate(mCUData.position, mCUData.positionOut, mCUData.velocity, dt, cudaGravity, cudaDimension, cudaPosition, mNumberOfParticles, mSimData);
	
	// sync the result back to the CPU
	// note: swapping pointers instead of copying only boosted the performance by 2fps@20,000 particles on a GTX1060
	float4* tmp = mCUData.position;
	mCUData.position = mCUData.positionOut;
	mCUData.positionOut = tmp;
	//CUDAERRORS(cudaMemcpy(mCUData.position, mCUData.positionOut, sizeof(ofVec4f) * mNumberOfParticles, cudaMemcpyDeviceToDevice));
	CUDAERRORS(cudaMemcpy(mParticlePosition, mCUData.position, sizeof(ofVec4f) * mNumberOfParticles, cudaMemcpyDeviceToHost));
}

void ParticleSystem::iUpdateThrust(float dt)
{
	// convert some host variables to device types
	float3 cudaGravity = make_float3(mGravity.x, mGravity.y, mGravity.z);
	float3 cudaDimension = make_float3(mDimension.x, mDimension.y, mDimension.z);

	float4* positionF4 = reinterpret_cast<float4*>(mParticlePosition);
	float4* velocityF4 = reinterpret_cast<float4*>(mParticleVelocity);

	//thrust::host_vector<float4> hostOut(mNumberOfParticles);

	ThrustHelper::thrustParticleUpdate(*mThrustData, positionF4, positionF4, velocityF4, dt, cudaGravity, cudaDimension, mNumberOfParticles, mSimData);

	//thrust::copy(hostOut.begin(), hostOut.end(), positionF4);
}

// debug function to count how many particles are outside the boundary
uint ParticleSystem::debug_testIfParticlesOutside()
{
	uint count = 0;
	for (uint i = 0; i < mNumberOfParticles; ++i)
	{
		ofVec3f particlePosition = mParticlePosition[i];
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

void ParticleSystem::toggleGenericSwitch()
{
	mGenericSwitch = !mGenericSwitch;
}