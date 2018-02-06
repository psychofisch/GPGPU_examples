#include "oclHelper.h"

namespace oclHelp
{
	cl::Context setupOpenCLContext(cl_uint platformId, cl_uint deviceId)
	{
		cl_int err = CL_SUCCESS;
		cl::Program program;
		std::vector<cl::Device> devices;
		cl::Context context;

		std::vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);
		if (platforms.size() == 0) {
			std::cout << "No OpenCL platforms available!\n";
			exit(1);
		}

		// create a context and get available devices
		cl::Platform platform = platforms[platformId]; // on a different machine, you may have to select a different platform
		std::cout << "Platform Name: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
		cl_context_properties properties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(platform)(), 0 };

		context = cl::Context(CL_DEVICE_TYPE_ALL, properties);

		devices = context.getInfo<CL_CONTEXT_DEVICES>();

		if (devices.size() == 0)
		{
			std::cout << "ERROR: no available OpenCL devices found\n";
			exit(1);
		}

		char deviceName[255];
		err = devices[deviceId].getInfo(CL_DEVICE_NAME, &deviceName);
		handle_clerror(err);
		std::cout << "using OpenCL device: " << deviceName << std::endl;

		return context;
	}

	void handle_clerror(cl_int err)
	{
		if (err != CL_SUCCESS) {
			std::cerr << "OpenCL Error: " << cl_errorstring(err) << std::string(".") << std::endl;
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
		case CL_INVALID_GLOBAL_WORK_SIZE:         return std::string("Invalid gloal work size");
			// case CL_INVALID_PROPERTY:                 return std::string("Invalid property");
		default:                                  return std::string("Unknown error code");
		}
	}
}