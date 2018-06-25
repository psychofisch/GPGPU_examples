#pragma once

#include <of3dPrimitives.h>
#include <ofColor.h>

#include "CollisionDefinitions.h"

class Cube :
	public ofBoxPrimitive
{
public:
	Cube();
	~Cube();

	ofColor mColor;

	// this method needs to be called everytime the objects gets rotated (TODO: overload rotate-functions?)
	void recalculateMinMax();
	// returns the local dimensions of the object
	MinMaxData getLocalMinMax() const;
	// adds the global position to the minMax values
	MinMaxData getGlobalMinMax() const;

private:
	MinMaxData mMinMax;
};
