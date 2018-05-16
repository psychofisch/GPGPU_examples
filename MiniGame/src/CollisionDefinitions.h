#pragma once

#include <ofVec3f.h>
#include <ofVec4f.h>

#define OUT 
typedef unsigned int uint;

namespace vec3
{
	static const ofVec3f up(0.f, 1.f, 0.f);
	static const ofVec3f forward(0.f, 0.f, 1.f);
	static const ofVec3f left(1.f, 0.f, 0.f);
}

struct MinMaxData
{
	ofVec4f min, max;

	MinMaxData operator+(ofVec3f p_)
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
	}
};

// intersection code from https://github.com/BSVino/MathForGameDevelopers/blob/line-box-intersection/math/collision.cpp

inline bool ClipLine(int d, const MinMaxData& aabbBox, const ofVec3f& v0, const ofVec3f& v1, float& f_low, float& f_high)
{
	// f_low and f_high are the results from all clipping so far. We'll write our results back out to those parameters.

	// f_dim_low and f_dim_high are the results we're calculating for this current dimension.
	float f_dim_low, f_dim_high;

	// Find the point of intersection in this dimension only as a fraction of the total vector http://youtu.be/USjbg5QXk3g?t=3m12s
	f_dim_low = (aabbBox.min[d] - v0[d]) / (v1[d] - v0[d]);
	f_dim_high = (aabbBox.max[d] - v0[d]) / (v1[d] - v0[d]);

	// Make sure low is less than high
	if (f_dim_high < f_dim_low)
		swap(f_dim_high, f_dim_low);

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
inline bool LineAABBIntersection(const MinMaxData& aabbBox, const ofVec3f& v0, const ofVec3f& v1, ofVec3f& vecIntersection, float& flFraction)
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
	ofVec3f b = v1 - v0;
	vecIntersection = v0 + b * f_low;

	flFraction = f_low;

	return true;
}


