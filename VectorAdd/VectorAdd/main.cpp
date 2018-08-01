#include <vector>
#include <random>
#include <iostream>
#include <fstream>
#include <string>
#include <CL\cl.hpp>

#include "GL\glew.h"
#include "GL\glut.h"
#include "GL\glfw3.h"

#include "thrust.h"

extern "C" void cudaVectorAdd(const int *vectorA, const int *vectorB, int *vectorC, int numElements);
void handle_clerror(cl_int err, int line);
std::string cl_errorstring(cl_int err);
void clVectorAdd(const std::vector<int>& vecA, const std::vector<int>& vecB, std::vector<int>& vecC);

#define DEBUG

#define GL_ERROR() {GLenum err; while ((err = glGetError()) != GL_NO_ERROR) std::cout << std::endl << __LINE__ << " " << gluErrorString(err) << std::endl;}

void main(int argc, char *argv[])
{
	//prep
	int arraySize = 10;
	std::vector<int> vectorA(arraySize);
	std::vector<int> vectorB(arraySize);
	std::vector<int> vectorC(arraySize);

	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_int_distribution<> intR(-100, 100);

	for (size_t i = 0; i < vectorA.size(); i++)
	{
#ifdef DEBUG
		vectorA[i] = -10;
		vectorB[i] = +10;
		vectorC[i] = 666;
#elif
		vectorA[i] = intR(mt);
		vectorB[i] = intR(mt);
#endif
	}
	//*** p

	if (strcmp("cpu", argv[1]) == 0)
	{
		//CPU
		std::cout << "CPU";
		for (size_t i = 0; i < arraySize; i++)
		{
			vectorC[i] = vectorA[i] + vectorB[i];
		}
		//*** cpu
	}
	else if (strcmp("glsl", argv[1]) == 0)
	{
		// Compute Shader
		std::cout << "Compute Shader";
		
		//*** cs
	}
	else if (strcmp("ocl", argv[1]) == 0)
	{
		// Compute Shader
		std::cout << "OpenCL";
		clVectorAdd(vectorA, vectorB, vectorC);
		//*** cs
	}
	else if (strcmp("cuda", argv[1]) == 0)
	{
		// CUDA
		std::cout << "CUDA";
		//allocate CUDA buffers
		cudaVectorAdd(vectorA.data(), vectorB.data(), vectorC.data(), arraySize);
		//*** cuda
	}
	else if (strcmp("thrust", argv[1]) == 0)
	{
		// CUDA
		std::cout << "Thrust";
		//allocate CUDA buffers
		thrustVectorAdd(vectorA.data(), vectorB.data(), vectorC.data(), arraySize);
		//*** cuda
	}
	else
	{
		std::cout << "unknown";
	}

	std::cout << " mode activated.\n";
#ifdef DEBUG
	//check
	bool check = true;
	for (size_t i = 0; i < vectorC.size(); i++)
	{
		if (vectorC[i] != 0)
		{
			check = false;
			break;
		}
	}

	std::cout << "Check? " << ((check) ? "PASSED" : "FAILED!") << std::endl;
#endif // DEBUG

	std::cin.ignore();
}

void clVectorAdd(const std::vector<int>& vecA, const std::vector<int>& vecB, std::vector<int>& vecC)
{
	cl_int err;
	cl::Context context;

	size_t platformId = 0;
	size_t deviceId = 0;

	// query all available platforms
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	if (platforms.size() == 0 || platforms.size() < platformId + 1) {
		std::cout << "ERROR: OpenCL platform not available!\n";
		return;
	}

	// create a context and get available devices
	cl::Platform platform = platforms[platformId];
	std::cout << "Platform Name: " << platform.getInfo<CL_PLATFORM_NAME>(&err) << std::endl;
	handle_clerror(err, __LINE__);
	cl_context_properties properties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(platform)(), 0 };

	// create context on desired platform and select device
	context = cl::Context(CL_DEVICE_TYPE_ALL, properties);
	std::vector<cl::Device> mDevices = context.getInfo<CL_CONTEXT_DEVICES>(&err);
	handle_clerror(err, __LINE__);

	if (mDevices.size() == 0 || mDevices.size() < deviceId + 1)
	{
		std::cout << "ERROR: no available OpenCL devices found\n";
		return;
	}

	char deviceName[255];
	err = mDevices[deviceId].getInfo(CL_DEVICE_NAME, &deviceName);
	std::cout << "using OpenCL device: " << deviceName << std::endl;

	cl::Program program;
	// load and build the kernel
	const char* file = "vectorAdd.cl";
	std::ifstream sourceFile(file);
	if (!sourceFile)
	{
		std::cout << "kernel source file \"" << file << "\" not found!" << std::endl;
		return;
	}
	std::string sourceCode(
		std::istreambuf_iterator<char>(sourceFile),
		(std::istreambuf_iterator<char>()));
	cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length() + 1));
	program = cl::Program(context, source);

	err = program.build(mDevices);
	if (err != CL_SUCCESS)
	{
		std::string s;
		program.getBuildInfo(mDevices[0], CL_PROGRAM_BUILD_LOG, &s);
		std::cout << s << std::endl;
		program.getBuildInfo(mDevices[0], CL_PROGRAM_BUILD_OPTIONS, &s);
		std::cout << s << std::endl;
		return;
	}
	handle_clerror(err, __LINE__);

	//create kernels
	cl::Kernel kernel = cl::Kernel(program, "vectorAdd", &err);
	handle_clerror(err, __LINE__);

	cl::CommandQueue queue = cl::CommandQueue(context, mDevices[deviceId], 0, &err);
	handle_clerror(err, __LINE__);

	//create and fill kernels
	size_t size = sizeof(int) * vecC.size();
	cl::Buffer vecs[3];
	auto it = vecA.begin();
	vecs[0] = cl::Buffer(context, CL_MEM_READ_ONLY, size, 0, &err);
	vecs[1] = cl::Buffer(context, CL_MEM_READ_ONLY, size, 0, &err);
	vecs[2] = cl::Buffer(context, CL_MEM_WRITE_ONLY, size, 0, &err);

	err = queue.enqueueWriteBuffer(vecs[0], CL_FALSE, 0, size, vecA.data());
	err = queue.enqueueWriteBuffer(vecs[1], CL_TRUE, 0, size, vecB.data());
	handle_clerror(err, __LINE__);

	err = kernel.setArg(0, vecs[0]);
	err = kernel.setArg(1, vecs[1]);
	err = kernel.setArg(2, vecs[2]);
	err = kernel.setArg(3, vecC.size());
	handle_clerror(err, __LINE__);

	//dispatch kernel
	// calculate the global and local size
	cl::NDRange local;
	cl::NDRange global;
	cl::NDRange offset(0);

	local = cl::NDRange(10);
	global = cl::NDRange(10);

	err = queue.enqueueNDRangeKernel(kernel, offset, global, local);
	handle_clerror(err, __LINE__);

	//get result from GPU
	err = queue.enqueueReadBuffer(vecs[2], CL_TRUE, 0, size, vecC.data());
	handle_clerror(err, __LINE__);

	return;
}

inline void handle_clerror(cl_int err, int line)
{
	if (err != CL_SUCCESS) {
		std::cerr << "OpenCL Error: " << cl_errorstring(err) << "\n on line " << line << std::endl;
		std::cin.ignore();
		exit(EXIT_FAILURE);
	}
}

std::string cl_errorstring(cl_int err) {
	switch (err) {
	case CL_SUCCESS:                          return std::string("Success");
	case CL_DEVICE_NOT_FOUND:                 return std::string("Device not found");
	case CL_DEVICE_NOT_AVAILABLE:             return std::string("Device not available");
	case CL_COMPILER_NOT_AVAILABLE:           return std::string("Compiler not available");
	case CL_MEM_OBJECT_ALLOCATION_FAILURE:    return std::string("Memory object allocation failure");
	case CL_OUT_OF_RESOURCES:                 return std::string("Out of resources");
	case CL_OUT_OF_HOST_MEMORY:               return std::string("Out of host memory");
	case CL_PROFILING_INFO_NOT_AVAILABLE:     return std::string("Profiling information not available");
	case CL_MEM_COPY_OVERLAP:                 return std::string("Memory copy overlap");
	case CL_IMAGE_FORMAT_MISMATCH:            return std::string("Image format mismatch");
	case CL_IMAGE_FORMAT_NOT_SUPPORTED:       return std::string("Image format not supported");
	case CL_BUILD_PROGRAM_FAILURE:            return std::string("Program build failure");
	case CL_MAP_FAILURE:                      return std::string("Map failure");
		// case CL_MISALIGNED_SUB_BUFFER_OFFSET:     return std::string("Misaligned sub buffer offset");
		// case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST: return std::string("Exec status error for events in wait list");
	case CL_INVALID_VALUE:                    return std::string("Invalid value");
	case CL_INVALID_DEVICE_TYPE:              return std::string("Invalid device type");
	case CL_INVALID_PLATFORM:                 return std::string("Invalid platform");
	case CL_INVALID_DEVICE:                   return std::string("Invalid device");
	case CL_INVALID_CONTEXT:                  return std::string("Invalid context");
	case CL_INVALID_QUEUE_PROPERTIES:         return std::string("Invalid queue properties");
	case CL_INVALID_COMMAND_QUEUE:            return std::string("Invalid command queue");
	case CL_INVALID_HOST_PTR:                 return std::string("Invalid host pointer");
	case CL_INVALID_MEM_OBJECT:               return std::string("Invalid memory object");
	case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:  return std::string("Invalid image format descriptor");
	case CL_INVALID_IMAGE_SIZE:               return std::string("Invalid image size");
	case CL_INVALID_SAMPLER:                  return std::string("Invalid sampler");
	case CL_INVALID_BINARY:                   return std::string("Invalid binary");
	case CL_INVALID_BUILD_OPTIONS:            return std::string("Invalid build options");
	case CL_INVALID_PROGRAM:                  return std::string("Invalid program");
	case CL_INVALID_PROGRAM_EXECUTABLE:       return std::string("Invalid program executable");
	case CL_INVALID_KERNEL_NAME:              return std::string("Invalid kernel name");
	case CL_INVALID_KERNEL_DEFINITION:        return std::string("Invalid kernel definition");
	case CL_INVALID_KERNEL:                   return std::string("Invalid kernel");
	case CL_INVALID_ARG_INDEX:                return std::string("Invalid argument index");
	case CL_INVALID_ARG_VALUE:                return std::string("Invalid argument value");
	case CL_INVALID_ARG_SIZE:                 return std::string("Invalid argument size");
	case CL_INVALID_KERNEL_ARGS:              return std::string("Invalid kernel arguments");
	case CL_INVALID_WORK_DIMENSION:           return std::string("Invalid work dimension");
	case CL_INVALID_WORK_GROUP_SIZE:          return std::string("Invalid work group size");
	case CL_INVALID_WORK_ITEM_SIZE:           return std::string("Invalid work item size");
	case CL_INVALID_GLOBAL_OFFSET:            return std::string("Invalid global offset");
	case CL_INVALID_EVENT_WAIT_LIST:          return std::string("Invalid event wait list");
	case CL_INVALID_EVENT:                    return std::string("Invalid event");
	case CL_INVALID_OPERATION:                return std::string("Invalid operation");
	case CL_INVALID_GL_OBJECT:                return std::string("Invalid OpenGL object");
	case CL_INVALID_BUFFER_SIZE:              return std::string("Invalid buffer size");
	case CL_INVALID_MIP_LEVEL:                return std::string("Invalid mip-map level");
	case CL_INVALID_GLOBAL_WORK_SIZE:         return std::string("Invalid global work size");
		// case CL_INVALID_PROPERTY:                 return std::string("Invalid property");
	default:                                  return std::string("Unknown error code");
	}
}
