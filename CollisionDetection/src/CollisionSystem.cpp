#include "CollisionSystem.h"



CollisionSystem::CollisionSystem()
{
}


CollisionSystem::~CollisionSystem()
{
}

void CollisionSystem::getCollisions(std::vector<Cube>& cubes, OUT std::vector<int>& collisions)
{
	if (cubes.size() != collisions.size())
	{
		std::cout << "CollisionSystem, " << __LINE__ << ": the input and output vector do not have the same size!\n";
	}

	std::vector<ofVec3f[2]> minMax(cubes.size());
	for (size_t i = 0; i < cubes.size(); ++i) // calculate bounding boxes
	{
		const Cube& currentCube = cubes[i];
		const std::vector<ofVec3f>& vertices = currentCube.getMesh().getVertices();
		ofVec3f min, max, pos;
		min = ofVec3f(INFINITY);
		max = ofVec3f(-INFINITY);
		pos = currentCube.getPosition();
		for (size_t o = 0; o < vertices.size(); o++)
		{
			ofVec3f current = vertices[o] + pos;
			for (size_t p = 0; p < 3; p++)
			{
				if (current[p] < min[p])
					min[p] = current[p];
				else if (current[p] > max[p])
					max[p] = current[p];
			}
		}
		minMax[i][0] = min;
		minMax[i][1] = max;
	}

	for (size_t i = 0; i < minMax.size(); i++)
	{
		ofVec3f currentMin = minMax[i][0];
		ofVec3f currentMax = minMax[i][1];
		int result = -1;
		for (size_t j = 0; j < minMax.size(); j++)
		{
			if (i == j)
				continue;
			int cnt = 0;
			for (size_t p = 0; p < 3; p++)
			{
				ofVec3f otherMin = minMax[j][0];
				ofVec3f otherMax = minMax[j][1];
				if ((otherMin[p] < currentMax[p] && otherMin[p] > currentMin[p])
					|| (otherMax[p] < currentMax[p] && otherMax[p] > currentMin[p])
					|| (otherMax[p] > currentMax[p] && otherMin[p] < currentMin[p])
					|| (otherMax[p] < currentMax[p] && otherMin[p] > currentMin[p]))
					cnt++;
			}

			if (cnt >= 3)
			{
				result = j;
				collisions[i] = result;
				break;
			}
			else
				collisions[i] = result;
		}
	}
}
