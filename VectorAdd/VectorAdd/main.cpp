#include <vector>
#include <random>
#include <iostream>

#include "thrust.h"

extern "C" void cudaVectorAdd(const int *vectorA, const int *vectorB, int *vectorC, int numElements);

#define DEBUG

void main(int argc, char *argv[])
{
	//prep
	int arraySize = 10;
	std::vector<int> vectorA(arraySize);
	std::vector<int> vectorB(arraySize);
	std::vector<int> vectorC(arraySize);

	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_int_distribution<> intR(-100, 100);

	for (size_t i = 0; i < vectorA.size(); i++)
	{
#ifdef DEBUG
		vectorA[i] = -10;
		vectorB[i] = +10;
		vectorC[i] = 666;
#elif
		vectorA[i] = intR(mt);
		vectorB[i] = intR(mt);
#endif
	}
	//*** p

	if (strcmp("cpu", argv[1]) == 0)
	{
		//CPU
		std::cout << "CPU";
		for (size_t i = 0; i < arraySize; i++)
		{
			vectorC[i] = vectorA[i] + vectorB[i];
		}
		//*** cpu
	}
	else if (strcmp("glsl", argv[1]) == 0)
	{
		// Compute Shader
		std::cout << "Compute Shader";

		//*** cs
	}
	else if (strcmp("cuda", argv[1]) == 0)
	{
		// CUDA
		std::cout << "CUDA";
		//allocate CUDA buffers
		cudaVectorAdd(vectorA.data(), vectorB.data(), vectorC.data(), arraySize);
		//*** cuda
	}
	else if (strcmp("thrust", argv[1]) == 0)
	{
		// CUDA
		std::cout << "Thrust";
		//allocate CUDA buffers
		thrustVectorAdd(vectorA.data(), vectorB.data(), vectorC.data(), arraySize);
		//*** cuda
	}
	else
	{
		std::cout << "unknown";
	}

	std::cout << " mode acitvated.\n";
#ifdef DEBUG
	//check
	bool check = true;
	for (size_t i = 0; i < vectorC.size(); i++)
	{
		if (vectorC[i] != 0)
		{
			check = false;
			break;
		}
	}

	std::cout << "Check? " << ((check) ? "PASSED" : "FAILED!") << std::endl;
#endif // DEBUG

	std::cin.ignore();
}