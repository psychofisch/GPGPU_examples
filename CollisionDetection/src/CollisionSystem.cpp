#include "CollisionSystem.h"

float4 make_float4(ofVec4f v_)
{
	float4 tmp;
	tmp.x = v_.x; tmp.y = v_.y; tmp.z = v_.z; tmp.w = v_.w;
	return tmp;
}

float3 make_float3(ofVec4f v_)
{
	float3 tmp;
	tmp.x = v_.x; tmp.y = v_.y; tmp.z = v_.z;
	return tmp;
}

CollisionSystem::CollisionSystem()
	:mMode(CPU)
{
}


CollisionSystem::~CollisionSystem()
{
}


void CollisionSystem::setupAll(ofxXmlSettings & settings)
{
	setupCPU(settings);
	setupCompute(settings);
	setupCUDA(settings);
	setupOCL(settings);
	setupThrust(settings);
}

void CollisionSystem::setupCPU(ofxXmlSettings & settings)
{
	mCPUThreshold = settings.getValue("CPU:THRESHOLD", 1000);

	mAvailableModes[ComputeMode::CPU] = settings.getValue("CPU:ENABLED", true);
}

void CollisionSystem::setupCompute(ofxXmlSettings & settings)
{
	// compile the compute code
	if (mComputeData.computeShader.setupShaderFromFile(GL_COMPUTE_SHADER, settings.getValue("COMPUTE:SOURCE", "collision.compute"))
		&& mComputeData.computeShader.linkProgram())
	{
		//allocate buffer memory
		//mComputeData.minMaxBuffer.allocate(sizeof(ofVec4f) * mCapacity, mPosition, GL_DYNAMIC_DRAW);

		mAvailableModes[ComputeMode::COMPUTE_SHADER] = settings.getValue("COMPUTE:ENABLED", true);
	}
	else
	{
		mAvailableModes[ComputeMode::COMPUTE_SHADER] = false;
	}
}
void CollisionSystem::setupCUDA(ofxXmlSettings & settings)
{
	//load CUDA command line arguments from settings file
	const int cmdArgc = settings.getValue("CUDA:ARGC", 0);
	const char* cmdArgs = settings.getValue("CUDA:ARGV", "").c_str();

	// find a CUDA device
	findCudaDevice(cmdArgc, &cmdArgs);

	mCudata.currentArraySize = 0;

	// "checkCudaErrors" will quit the program in case of a problem, so it is safe to assume that if the program reached this point CUDA will work
	mAvailableModes[ComputeMode::CUDA] = settings.getValue("CUDA:ENABLED", true);
}

void CollisionSystem::setupOCL(ofxXmlSettings & settings)
{
	int platformID = settings.getValue("OCL:PLATFORMID", 0);
	int deviceID = settings.getValue("OCL:DEVICEID", 0);
	std::string sourceFile = settings.getValue("OCL:SOURCE", "data/collision.cl");

	// try to set up an OpenCL context
	if (!mOCLHelper.setupOpenCLContext(platformID, deviceID))
	{
		// try to compile the source file
		if (!mOCLHelper.compileKernel(sourceFile.c_str()))
		{
			// if all of the above worked -> set up all buffers and settings
			cl::Context context = mOCLHelper.getCLContext();

			// query the maximum work group size rom the device
			cl_int err = mOCLHelper.getDevice().getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &mOCLData.maxWorkGroupSize);
			mOCLData.maxWorkGroupSize /= 2;//testing showed best performance at half the max ThreadCount
			oclHelper::handle_clerror(err, __LINE__);

			mOCLData.currentArraySize = 0;

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

void CollisionSystem::setupThrust(ofxXmlSettings & settings)
{
	mAvailableModes[THRUST] = settings.getValue("THRUST:ENABLED", true);
}

CollisionSystem::ComputeMode CollisionSystem::getMode() const
{
	return mMode;
}

CollisionSystem::ComputeMode CollisionSystem::nextMode(CollisionSystem::ComputeMode current) const
{
	int enumSize = ComputeMode::COMPUTEMODES_SIZE;

	for (int i = 0; i < enumSize; ++i)
	{
		int next = (current + 1 + i) % enumSize;
		if (mAvailableModes[next]) // checks if the next mode is even available
			return static_cast<ComputeMode>(next);
	}

	return current;
}

void CollisionSystem::setMode(ComputeMode m)
{
	mMode = m;
}

void CollisionSystem::getCollisions(std::vector<Cube>& cubes, OUT std::vector<int>& collisions)
{
	switch (mMode)
	{
		case CPU: iGetCollisionsCPU(cubes, collisions);
			break;
		case COMPUTE_SHADER: iGetCollisionsCompute(cubes, collisions);
			break;
		case CUDA: iGetCollisionsCUDA(cubes, collisions);
			break;
		case OPENCL: iGetCollisionsOCL(cubes, collisions);
			break;
		case THRUST: iGetCollisionsThrust(cubes, collisions);
			break;
		default:
			break;
	}
}

void CollisionSystem::iGetCollisionsCPU(std::vector<Cube>& cubes, OUT std::vector<int>& collisions)
{
	if (cubes.size() != collisions.size())
	{
		std::cout << "CollisionSystem, " << __LINE__ << ": the input and output vector do not have the same size!\n";
	}

	std::vector<MinMaxData> mMinMax(cubes.size());// OPT: storing mMinMax locally is about 10% faster than having a member
	// read bounding boxes
	for (int i = 0; i < cubes.size(); ++i)
	{
		mMinMax[i] = cubes[i].getGlobalMinMax();
	}

	// check min and max of all boxes for collision
	std::vector<int> tmpCol(collisions.size());// OPT: local storage has a bit better performance again
	for (int i = 0; i < mMinMax.size(); i++) 
	{
		const ofVec3f currentMin = mMinMax[i].min;
		const ofVec3f currentMax = mMinMax[i].max;
		//ofVec3f otherMin, otherMax;
		int result = -1;
		for (int j = 0; j < mMinMax.size(); j++)
		{
			if (i == j)
				continue;

			const ofVec3f otherMin = mMinMax[j].min;
			const ofVec3f otherMax = mMinMax[j].max;

			if ((  (otherMin.x < currentMax.x && otherMin.x > currentMin.x)
				|| (otherMax.x < currentMax.x && otherMax.x > currentMin.x)
				|| (otherMax.x > currentMax.x && otherMin.x < currentMin.x)
				|| (otherMax.x < currentMax.x && otherMin.x > currentMin.x))
			&&
				(  (otherMin.z < currentMax.z && otherMin.z > currentMin.z)
				|| (otherMax.z < currentMax.z && otherMax.z > currentMin.z)
				|| (otherMax.z > currentMax.z && otherMin.z < currentMin.z)
				|| (otherMax.z < currentMax.z && otherMin.z > currentMin.z))
			&&	
				(  (otherMin.y < currentMax.y && otherMin.y > currentMin.y)
				|| (otherMax.y < currentMax.y && otherMax.y > currentMin.y)
				|| (otherMax.y > currentMax.y && otherMin.y < currentMin.y)
				|| (otherMax.y < currentMax.y && otherMin.y > currentMin.y))
			) // TODO: optimize this
			{
				result = j;
				break;// OPT: do not delete this (30% performance loss)
			}
		}

		tmpCol[i] = result;
	}

	collisions = tmpCol;
}

void CollisionSystem::iGetCollisionsCompute(std::vector<Cube>& cubes, OUT std::vector<int>& collisions)
{
	if (cubes.size() != collisions.size())
	{
		std::cout << "CollisionSystem, " << __LINE__ << ": the input and output vector do not have the same size!\n";
	}

	if (mComputeData.minMaxBuffer.size() < sizeof(MinMaxData) * cubes.size())
	{
		mComputeData.minMaxBuffer.allocate(sizeof(MinMaxData) * cubes.size(), GL_DYNAMIC_DRAW);
	}

	if (mComputeData.collisionBuffer.size() < sizeof(int) * cubes.size())
	{
		mComputeData.collisionBuffer.allocate(sizeof(int) * cubes.size(), GL_DYNAMIC_DRAW);
	}

	std::vector<MinMaxData> mMinMax(cubes.size());// OPT: storing mMinMax locally is about 10% faster than having a member
	// read bounding boxes
	for (int i = 0; i < cubes.size(); ++i)
	{
		MinMaxData currentCube = cubes[i].getGlobalMinMax();
		mMinMax[i].min = currentCube.min;
		mMinMax[i].max = currentCube.max;
	}

	// copy minMax data to GPU
	mComputeData.minMaxBuffer.updateData(mMinMax);

	mComputeData.minMaxBuffer.bindBase(GL_SHADER_STORAGE_BUFFER, 0);
	mComputeData.collisionBuffer.bindBase(GL_SHADER_STORAGE_BUFFER, 1);

	mComputeData.computeShader.begin();
	mComputeData.computeShader.setUniform1i("amountOfCubes", cubes.size());

	// call the kernel
	// local size: hard-coded "512", because it is also hard-coded in the kernel source code
	mComputeData.computeShader.dispatchCompute(std::ceilf(float(cubes.size()) / 512), 1, 1);
	mComputeData.computeShader.end();//forces the program to wait until the calculation is finished 

	// copy the GPU data back to the CPU
	int* collisionsGPU = mComputeData.collisionBuffer.map<int>(GL_READ_ONLY);
	memcpy(&collisions[0], collisionsGPU, sizeof(int) * collisions.size());
	mComputeData.collisionBuffer.unmap();
}

void CollisionSystem::iGetCollisionsCUDA(std::vector<Cube>& cubes, OUT std::vector<int>& collisions)
{
	if (cubes.size() != collisions.size())
	{
		std::cout << "CollisionSystem, " << __LINE__ << ": the input and output vector do not have the same size!\n";
	}

	if (mCudata.currentArraySize < cubes.size())
	{
		if (mCudata.currentArraySize > 0)
		{
			cudaFree(mCudata.minMaxBuffer);
			cudaFree(mCudata.collisionBuffer);
		}

		mCudata.currentArraySize = cubes.size();

		std::cout << "CUDA: allocating memory for " << mCudata.currentArraySize << " cubes.\n";

		checkCudaErrors(cudaMallocManaged(&mCudata.minMaxBuffer, sizeof(float4) * 2 * mCudata.currentArraySize));
		checkCudaErrors(cudaMallocManaged(&mCudata.collisionBuffer, sizeof(int) * mCudata.currentArraySize));
	}

	float4* minMax = mCudata.minMaxBuffer;
	// read bounding boxes
	for (int i = 0; i < cubes.size(); ++i)
	{
		MinMaxData currentCube = cubes[i].getGlobalMinMax();
		minMax[(i * 2)] = make_float4(currentCube.min);
		minMax[(i * 2) +1] = make_float4(currentCube.max);
	}

	cudaDeviceSynchronize();

	// calculate the collisions
	cudaGetCollisions(mCudata.minMaxBuffer, mCudata.collisionBuffer, mCudata.currentArraySize);

	cudaDeviceSynchronize();

	// copy the collision data from the local host buffer to the provided vector
	memcpy(&collisions[0], mCudata.collisionBuffer, sizeof(int) * collisions.size());
}

void CollisionSystem::iGetCollisionsOCL(std::vector<Cube>& cubes, OUT std::vector<int>& collisions)
{
	if (cubes.size() != collisions.size())
	{
		std::cout << "CollisionSystem, " << __LINE__ << ": the input and output vector do not have the same size!\n";
	}

	std::vector<float4> minMax(cubes.size() * 2);
	// read bounding boxes
	for (int i = 0; i < cubes.size(); ++i)
	{
		MinMaxData currentCube = cubes[i].getGlobalMinMax();
		minMax[(i * 2)] = make_float4(currentCube.min);
		minMax[(i * 2) + 1] = make_float4(currentCube.max);
	}

	if (mOCLData.currentArraySize < cubes.size())
	{
		mOCLData.currentArraySize = cubes.size();

		std::cout << "OpenCL: allocating memory for " << mOCLData.currentArraySize << " cubes.\n";

		cl_int err;
		mOCLData.minMaxBuffer = cl::Buffer(mOCLHelper.getCLContext(), CL_MEM_READ_WRITE, sizeof(float4) * 2 * mOCLData.currentArraySize, 0, &err);
		oclHelper::handle_clerror(err, __LINE__);
		mOCLData.collisionBuffer = cl::Buffer(mOCLHelper.getCLContext(), CL_MEM_READ_WRITE, sizeof(int) * mOCLData.currentArraySize, 0, &err);
		oclHelper::handle_clerror(err, __LINE__);
	}

	cl_int err;
	cl::CommandQueue queue = mOCLHelper.getCommandQueue();

	// copy the minMaxBuffer to the GPU
	err = queue.enqueueWriteBuffer(mOCLData.minMaxBuffer, CL_FALSE, 0, sizeof(float4) * 2 * mOCLData.currentArraySize, &minMax[0]);
	oclHelper::handle_clerror(err, __LINE__);

	cl::Kernel kernel = mOCLHelper.getKernel();

	// set all the kernel arguments
	err = kernel.setArg(0, mOCLData.minMaxBuffer);
	oclHelper::handle_clerror(err, __LINE__);
	err = kernel.setArg(1, mOCLData.collisionBuffer);
	oclHelper::handle_clerror(err, __LINE__);
	err = kernel.setArg(2, mOCLData.currentArraySize);
	oclHelper::handle_clerror(err, __LINE__);

	// calculate the global and local size
	cl::NDRange local;
	cl::NDRange global;
	cl::NDRange offset(0);

	if (mOCLData.currentArraySize < mOCLData.maxWorkGroupSize)
	{
		local = cl::NDRange(mOCLData.currentArraySize);
		global = cl::NDRange(mOCLData.currentArraySize);
	}
	else
	{
		size_t f = std::ceilf(float(mOCLData.currentArraySize) / mOCLData.maxWorkGroupSize);
		local = cl::NDRange(mOCLData.maxWorkGroupSize);
		global = cl::NDRange(mOCLData.maxWorkGroupSize * f);
	}

	// call the kernel
	err = queue.enqueueNDRangeKernel(kernel, offset, global, local);
	oclHelper::handle_clerror(err, __LINE__);

	// copy the result back to the CPU
	err = queue.enqueueReadBuffer(mOCLData.collisionBuffer, CL_TRUE, 0, mOCLData.currentArraySize * sizeof(int), &collisions[0]);
	oclHelper::handle_clerror(err, __LINE__);
}

void CollisionSystem::iGetCollisionsThrust(std::vector<Cube>& cubes, OUT std::vector<int>& collisions)
{
	if (cubes.size() != collisions.size())
	{
		std::cout << "CollisionSystem, " << __LINE__ << ": the input and output vector do not have the same size!\n";
	}

	thrust::host_vector<ThrustHelper::MinMaxDataThrust> mMinMax(cubes.size());// OPT: storing mMinMax locally is about 10% faster than having a member
	// read bounding boxes
	for (int i = 0; i < cubes.size(); ++i)
	{
		MinMaxData currentCube = cubes[i].getGlobalMinMax();
		mMinMax[i].min = make_float4(currentCube.min);
		mMinMax[i].max = make_float4(currentCube.max);
	}

	// check min and max of all boxes for collision
	ThrustHelper::thrustGetCollisions(mThrustData, mMinMax.data(), collisions.data(), collisions.size());
}
