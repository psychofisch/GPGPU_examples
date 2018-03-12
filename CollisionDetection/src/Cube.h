#pragma once

#include <of3dPrimitives.h>
#include <ofColor.h>

class Cube :
	public ofBoxPrimitive
{
public:
	Cube();
	~Cube();

	ofColor mColor;

private:
};

