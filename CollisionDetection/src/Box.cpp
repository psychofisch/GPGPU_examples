#include "Box.h"



Box::Box()
{
}


Box::~Box()
{
}

void Box::recalculateMinMax()
{
	const std::vector<ofVec3f>& vertices = this->getMesh().getVertices();
	ofVec3f min, max, pos;
	min = ofVec3f(INFINITY);
	max = ofVec3f(-INFINITY);
	for (int o = 0; o < vertices.size(); o++)
	{
		ofVec3f current = vertices[o];
		for (int p = 0; p < 3; p++)
		{
			if (current[p] < min[p])
				min[p] = current[p];
			else if (current[p] > max[p])
				max[p] = current[p];
		}
	}

	mMinMax.min = min;
	mMinMax.max = max;
}

MinMaxData Box::getLocalMinMax() const
{
	return mMinMax;
}

MinMaxData Box::getGlobalMinMax() const
{
	return mMinMax + this->getPosition();
}
