#include <CL/cl.hpp>
#include <iostream>

namespace oclHelp
{
	cl::Context setupOpenCLContext(cl_uint platformId = 0, cl_uint deviceId = 0);
	void handle_clerror(cl_int err);
	std::string cl_errorstring(cl_int err);
}