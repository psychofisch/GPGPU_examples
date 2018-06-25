#pragma once

#include <ofVec3f.h>
#include <ofVec4f.h>

namespace vec3
{
	static const ofVec3f up(0.f, 1.f, 0.f);
	static const ofVec3f forward(0.f, 0.f, 1.f);
	static const ofVec3f left(1.f, 0.f, 0.f);
}

struct MinMaxData
{
	ofVec4f min, max;

	MinMaxData operator+(ofVec4f p_)
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


