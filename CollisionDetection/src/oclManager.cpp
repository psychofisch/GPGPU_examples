/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <oclUtils.h>

#include <iostream>
#include <assert.h>

namespace oclManager {
	////////////////////////////////////////////////////////////////////////////////
	// Sort of API-independent interface
	////////////////////////////////////////////////////////////////////////////////
	cl_platform_id cpPlatform;
	cl_context cxGPUContext;
	cl_command_queue cqCommandQueue;

	//Context initialization/deinitialization
	extern "C" void startupOpenCL(cl_uint uiTargetDevice = 0) {

		cl_platform_id cpPlatform;
		cl_uint uiNumDevices;
		//cl_uint uiTargetDevice = 0;
		cl_device_id* cdDevices;
		cl_int ciErrNum;

		// Get the NVIDIA platform
		std::cout << "oclGetPlatformID...\n\n";
		ciErrNum = oclGetPlatformID(&cpPlatform);
		oclCheckError(ciErrNum, CL_SUCCESS);

		// Get the devices
		std::cout << "clGetDeviceIDs...\n\n";
		ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 0, NULL, &uiNumDevices);
		oclCheckError(ciErrNum, CL_SUCCESS);
		cdDevices = (cl_device_id *)malloc(uiNumDevices * sizeof(cl_device_id));

		ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, uiNumDevices, cdDevices, NULL);
		oclCheckError(ciErrNum, CL_SUCCESS);

		// Set target device and Query number of compute units on uiTargetDevice
		uiTargetDevice = CLAMP(uiTargetDevice, 0, (uiNumDevices - 1));

		std::cout << "Using Device %u, " << uiTargetDevice << std::endl;
		oclPrintDevName(LOGBOTH, cdDevices[uiTargetDevice]);

		// Create the context
		std::cout << "\n\nclCreateContext...\n\n";
		cxGPUContext = clCreateContext(0, 1, &cdDevices[uiTargetDevice], NULL, NULL, &ciErrNum);
		oclCheckError(ciErrNum, CL_SUCCESS);

		//Create a command-queue
		std::cout << "clCreateCommandQueue...\n\n";
		cqCommandQueue = clCreateCommandQueue(cxGPUContext, cdDevices[uiTargetDevice], 0, &ciErrNum);
		oclCheckError(ciErrNum, CL_SUCCESS);

		free(cdDevices);
	}

	extern "C" void shutdownOpenCL(void) {
		cl_int ciErrNum;
		ciErrNum = clReleaseCommandQueue(cqCommandQueue);
		ciErrNum |= clReleaseContext(cxGPUContext);
		oclCheckError(ciErrNum, CL_SUCCESS);
	}

	//GPU buffer allocation
	extern "C" void allocateArray(cl_mem *memObj, size_t size) {
		cl_int ciErrNum;
		std::cout << " clCreateBuffer (GPU GMEM, %u bytes)...\n\n" << size << std::endl;
		*memObj = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, size, NULL, &ciErrNum);
		oclCheckError(ciErrNum, CL_SUCCESS);
	}

	extern "C" void freeArray(cl_mem memObj) {
		cl_int ciErrNum;
		ciErrNum = clReleaseMemObject(memObj);
		oclCheckError(ciErrNum, CL_SUCCESS);
	}

	//host<->device memcopies
	extern "C" void copyArrayFromDevice(void *hostPtr, cl_mem memObj, unsigned int vbo, size_t size) {
		cl_int ciErrNum;
		assert(vbo == 0);
		ciErrNum = clEnqueueReadBuffer(cqCommandQueue, memObj, CL_TRUE, 0, size, hostPtr, 0, NULL, NULL);
		oclCheckError(ciErrNum, CL_SUCCESS);
	}

	extern "C" void copyArrayToDevice(cl_mem memObj, const void *hostPtr, size_t offset, size_t size) {
		cl_int ciErrNum;
		ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, memObj, CL_TRUE, 0, size, hostPtr, 0, NULL, NULL);
		oclCheckError(ciErrNum, CL_SUCCESS);
	}
}
