#pragma once

#include "Cube.h"

class CollisionSystem
{
public:
	enum ComputeMode
	{
		CPU = 0,
		COMPUTE_SHADER,
		CUDA,
		OPENCL,
		THRUST,
		COMPUTEMODES_SIZE // these values are used as array indices, dont delete this!
	};

	CollisionSystem();
	~CollisionSystem();

	void getCollisions(std::vector<Cube>& cubes, OUT std::vector<int>& collisions);
private:
	
};

