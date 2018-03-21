#pragma once

#include <ofVec4f.h>

#define OUT 
typedef unsigned int uint;

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
};
