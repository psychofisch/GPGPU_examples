#include "CollisionSystem.h"



CollisionSystem::CollisionSystem()
	:mMode(CPU)
{
}


CollisionSystem::~CollisionSystem()
{
}


//void CollisionSystem::setupAll(ofxXmlSettings & settings)
//{
//	setupCPU(settings);
//	setupCompute(settings);
//	setupCUDA(settings);
//	setupOCL(settings);
//	//setupThrust(settings);
//}
//
//void CollisionSystem::setupCPU(ofxXmlSettings & settings)
//{
//	//mThreshold = settings.getValue("CPU:THRESHOLD", 1000);
//
//	mAvailableModes[ComputeMode::CPU] = settings.getValue("CPU:ENABLED", true);
//}

//void CollisionSystem::setupCompute(ofxXmlSettings & settings)
//{
//	// compile the compute code
//	if (mComputeData.computeShader.setupShaderFromFile(GL_COMPUTE_SHADER, settings.getValue("COMPUTE:SOURCE", "particles.compute"))
//		&& mComputeData.computeShader.linkProgram())
//	{
//		//allocate buffer memory
//		mComputeData.positionOutBuffer.allocate(sizeof(ofVec4f) * mCapacity, mPosition, GL_DYNAMIC_DRAW);
//		mComputeData.positionBuffer.allocate(sizeof(ofVec4f) * mCapacity, mPosition, GL_DYNAMIC_DRAW);
//		mComputeData.velocityBuffer.allocate(sizeof(ofVec4f) * mCapacity, mVelocity, GL_DYNAMIC_DRAW);
//
//		mAvailableModes[ComputeMode::COMPUTE_SHADER] = settings.getValue("COMPUTE:ENABLED", true);
//	}
//	else
//	{
//		mAvailableModes[ComputeMode::COMPUTE_SHADER] = false;
//	}
//}
//
//void CollisionSystem::setupCUDA(ofxXmlSettings & settings)
//{
//	//load CUDA command line arguments from settings file
//	const int cmdArgc = settings.getValue("CUDA:ARGC", 0);
//	const char* cmdArgs = settings.getValue("CUDA:ARGV", "").c_str();
//
//	// find a CUDA device
//	findCudaDevice(cmdArgc, &cmdArgs);
//
//	// allocate memory
//	// note: do not use the CUDAERRORS macro here, because the case when these allocations fail in release mode has to be handled
//	checkCudaErrors(cudaMalloc(&mCUData.position, sizeof(ofVec4f) * mCapacity));
//	checkCudaErrors(cudaMalloc(&mCUData.velocity, sizeof(ofVec4f) * mCapacity));
//	checkCudaErrors(cudaMalloc(&mCUData.positionOut, sizeof(ofVec4f) * mCapacity));
//
//	// "checkCudaErrors" will quit the program in case of a problem, so it is safe to assume that if the program reached this point CUDA will work
//	mAvailableModes[ComputeMode::CUDA] = settings.getValue("CUDA:ENABLED", true);
//}
//
//void CollisionSystem::setupOCL(ofxXmlSettings & settings)
//{
//	int platformID = settings.getValue("OCL:PLATFORMID", 0);
//	int deviceID = settings.getValue("OCL:DEVICEID", 0);
//	std::string sourceFile = settings.getValue("OCL:SOURCE", "data/particles.cl");
//
//	// try to set up an OpenCL context
//	if (!mOCLHelper.setupOpenCLContext(platformID, deviceID))
//	{
//		// try to compile the source file
//		if (!mOCLHelper.compileKernel(sourceFile.c_str()))
//		{
//			// if all of the above worked -> set up all buffers and settings
//			cl::Context context = mOCLHelper.getCLContext();
//
//			// create buffers on the OpenCL device (don't need to be the GPU)
//			mOCLData.positionBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(ofVec4f) * mCapacity);
//			mOCLData.positionOutBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(ofVec4f) * mCapacity);
//			mOCLData.velocityBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(ofVec4f) * mCapacity);
//
//			// query the maximum work group size rom the device
//			cl_int err = mOCLHelper.getDevice().getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &mOCLData.maxWorkGroupSize);
//			mOCLData.maxWorkGroupSize /= 2;//testing showed best performance at half the max ThreadCount
//			oclHelper::handle_clerror(err, __LINE__);
//
//			mAvailableModes[ComputeMode::OPENCL] = settings.getValue("OCL:ENABLED", true);
//		}
//		else
//		{
//			std::cout << "ERROR: Unable to compile \"" << sourceFile << "\".\n";
//			mAvailableModes[ComputeMode::OPENCL] = false;
//		}
//	}
//	else
//	{
//		std::cout << "ERROR: Unable to create OpenCL context\n";
//		mAvailableModes[ComputeMode::OPENCL] = false;
//	}
//}
//
//void CollisionSystem::setupThrust(ofxXmlSettings & settings)
//{
//	/*mThrustData.position = thrust::device_malloc<float4>(mCapacity);
//	mThrustData.positionOut = thrust::device_malloc<float4>(mCapacity);
//	mThrustData.velocity = thrust::device_malloc<float4>(mCapacity);*/
//
//	mThrustData = ThrustHelper::setup(mNumberOfParticles);
//
//	mAvailableModes[ComputeMode::THRUST] = true;
//}

void CollisionSystem::getCollisions(std::vector<Cube>& cubes, OUT std::vector<int>& collisions)
{
	switch (mMode)
	{
		case CPU: iGetCollisionsCPU(cubes, collisions);
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

	/*if(mMinMax.size() != cubes.size())
		mMinMax.resize(cubes.size());*/

	std::vector<MinMaxData> mMinMax(cubes.size());//storing mMinMax locally is about 10% faster than having a member
	for (int i = 0; i < cubes.size(); ++i) // calculate bounding boxes
	{
		const Cube& currentCube = cubes[i];
		const std::vector<ofVec3f>& vertices = currentCube.getMesh().getVertices();
		ofVec3f min, max, pos;
		min = ofVec3f(INFINITY);
		max = ofVec3f(-INFINITY);
		pos = currentCube.getPosition();
		for (int o = 0; o < vertices.size(); o++)
		{
			ofVec3f current = vertices[o] + pos;
			for (int p = 0; p < 3; p++)
			{
				if (current[p] < min[p])
					min[p] = current[p];
				else if (current[p] > max[p])
					max[p] = current[p];
			}
		}
		mMinMax[i].min = min;
		mMinMax[i].max = max;
	}

	for (int i = 0; i < mMinMax.size(); i++)
	{
		ofVec3f currentMin = mMinMax[i].min;
		ofVec3f currentMax = mMinMax[i].max;
		int result = -1;
		for (int j = 0; j < mMinMax.size(); j++)
		{
			if (i == j)
				continue;
			int cnt = 0;
			ofVec3f otherMin = mMinMax[j].min;
			ofVec3f otherMax = mMinMax[j].max;
			bool loop = true;
			while (loop)
			{
				if (   (otherMin[cnt] < currentMax[cnt] && otherMin[cnt] > currentMin[cnt])
					|| (otherMax[cnt] < currentMax[cnt] && otherMax[cnt] > currentMin[cnt])
					|| (otherMax[cnt] > currentMax[cnt] && otherMin[cnt] < currentMin[cnt])
					|| (otherMax[cnt] < currentMax[cnt] && otherMin[cnt] > currentMin[cnt]))
				{
					loop = true;
					++cnt;
				}
				else
				{
					loop = false;
				}
			}

			if (cnt >= 3)
			{
				result = j;
				collisions[i] = result;
				break;
			}
			else
				collisions[i] = result;
		}
	}
}
