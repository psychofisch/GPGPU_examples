#include <CL/cl.hpp>
#include <iostream>
#include <fstream>
//#include <string>

class oclHelper
{
	public:
		oclHelper();
		bool setupOpenCLContext(cl_uint platformId = 0, cl_uint deviceId = 0);
		cl::Context& getCLContext();
		bool compileKernel(const char* file);

		static void handle_clerror(cl_int err);
		static std::string cl_errorstring(cl_int err);

	private:
		cl::Context mContext;
		cl::Kernel mKernel;
		cl::CommandQueue mQueue;
		std::vector<cl::Device> mDevices;
		size_t mDeviceId;
		size_t mGlobalSize;
		cl::Platform mPlatform;
		cl_int mError;
};
