#pragma once

#include "ParticleDefinitions.h"

#include <thrust\device_ptr.h>
#include <thrust\transform.h>
#include <thrust\iterator\zip_iterator.h>
#include <thrust\copy.h>

#include <helper_math.h>

namespace ThrustHelper
{
	struct InvertFunctor : thrust::binary_function < float3, float3, float3>;
}
