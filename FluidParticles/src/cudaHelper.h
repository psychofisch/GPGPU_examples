#ifndef CUDA_HELPER
#define CUDA_HELPER

// CUDA runtime
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>
#include <helper_math.h>

struct MinMaxDataCuda
{
	float4 min, max;

	/*MinMaxData operator+(float4 p_)
	{
	MinMaxData tmp;
	tmp.min = this->min + p_;
	tmp.max = this->max + p_;
	return tmp;
	}

	MinMaxData operator=(const MinMaxData& other)
	{
	this->min = other.min;
	this->max = other.max;
	return *this;
	}*/
};

inline __device__ __host__ float dim(const float3& v, size_t i)
{
	switch (i)
	{
	case 0: return v.x;//no breaks required
	case 1:	return v.y;
	case 2:	return v.z;
	}

	return NAN;
}

inline __device__ __host__ float& dim(float3& v, size_t i)
{
	switch (i)
	{
	case 0: return v.x;//no breaks required
	case 1:	return v.y;
	case 2:	return v.z;
	}

	return v.x;//TODO: return something that indicates an error
}

inline __device__ __host__ float dim(const float4& v, size_t i)
{
	switch (i)
	{
	case 0: return v.x;//no breaks required
	case 1:	return v.y;
	case 2:	return v.z;
	case 3: return v.w;
	}

	return NAN;
}

inline __device__ __host__ bool ClipLine(int d, const MinMaxDataCuda aabbBox, const float3 v0, const float3 v1, float& f_low, float& f_high)
{
	// f_low and f_high are the results from all clipping so far. We'll write our results back out to those parameters.

	// f_dim_low and f_dim_high are the results we're calculating for this current dimension.
	float f_dim_low, f_dim_high;

	// Find the point of intersection in this dimension only as a fraction of the total vector http://youtu.be/USjbg5QXk3g?t=3m12s
	f_dim_low = (dim(aabbBox.min, d) - dim(v0, d)) / (dim(v1, d) - dim(v0, d));
	f_dim_high = (dim(aabbBox.max, d) - dim(v0, d)) / (dim(v1, d) - dim(v0, d));

	// Make sure low is less than high
	if (f_dim_high < f_dim_low)
	{
		float tmp = f_dim_high;
		f_dim_high = f_dim_low;
		f_dim_low = tmp;
	}

	// If this dimension's high is less than the low we got then we definitely missed. http://youtu.be/USjbg5QXk3g?t=7m16s
	if (f_dim_high < f_low)
		return false;

	// Likewise if the low is less than the high.
	if (f_dim_low > f_high)
		return false;

	// Add the clip from this dimension to the previous results http://youtu.be/USjbg5QXk3g?t=5m32s
	f_low = max(f_dim_low, f_low);
	f_high = min(f_dim_high, f_high);

	if (f_low > f_high)
		return false;

	return true;
}

// Find the intersection of a line from v0 to v1 and an axis-aligned bounding box http://www.youtube.com/watch?v=USjbg5QXk3g
inline __device__ __host__ bool LineAABBIntersection(const MinMaxDataCuda aabbBox, const float3 v0, const float3 v1, float3& vecIntersection, float& flFraction)
{
	float f_low = 0;
	float f_high = 1;

	if (!ClipLine(0, aabbBox, v0, v1, f_low, f_high))
		return false;

	if (!ClipLine(1, aabbBox, v0, v1, f_low, f_high))
		return false;

	if (!ClipLine(2, aabbBox, v0, v1, f_low, f_high))
		return false;

	// The formula for I: http://youtu.be/USjbg5QXk3g?t=6m24s
	float3 b = v1 - v0;
	vecIntersection = v0 + b * f_low;

	flFraction = f_low;

	return true;
}

#endif
