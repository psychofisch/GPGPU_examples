#include <vector>
#include <random>
#include <iostream>
#include <string>

#include "GL\glew.h"
#include "GL\glut.h"
#include "GL\glfw3.h"

#include "thrust.h"

extern "C" void cudaVectorAdd(const int *vectorA, const int *vectorB, int *vectorC, int numElements);

#define DEBUG

#define GL_ERROR() {GLenum err; while ((err = glGetError()) != GL_NO_ERROR) std::cout << std::endl << __LINE__ << " " << gluErrorString(err) << std::endl;}

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
		glfwInit();
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 4);
		glfwWindowHint(GLFW_VISIBLE, false);
		GLFWwindow* window = glfwCreateWindow(800, 600, "GLFW test", NULL, NULL);
		glfwMakeContextCurrent(window);

		glewInit();

		const char* shaderSource = "#version 430"
			"layout(std140, binding = 0) buffer VectorA {int vecA[];};"
			"layout(std140, binding = 1) buffer VectorB {int vecB[];};"
			"layout(std140, binding = 2) buffer VectorC {int vecC[];};"
			"uniform int numberOfElements;"
			"layout(local_size_x = 512, local_size_y = 1, local_size_z = 1) in;"
			"void main() {"
			"uint index = gl_GlobalInvocationID.x;"
			"if(index < numberOfElements)"
			"vecC[index]=vecA[index]+vecB[index];"
			"}";
		
		GLint shaderLength = strlen(shaderSource);
		GLuint glShader = glCreateShader(GL_COMPUTE_SHADER);
		glShaderSource(glShader, 1, &shaderSource, &shaderLength);
		glCompileShader(glShader);
		GLuint vectorAddProgram = glCreateProgram();
		glAttachShader(vectorAddProgram, glShader);
		GL_ERROR();
		glLinkProgram(vectorAddProgram);
		
		glUseProgram(vectorAddProgram);
		GL_ERROR();

		GLuint vecs[3];
		glCreateBuffers(3, vecs);

		glBindBuffer(GL_SHADER_STORAGE_BUFFER, vecs[0]);
		glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(int) * arraySize, vectorA.data(), GL_DYNAMIC_DRAW);
		GL_ERROR();

		glBindBuffer(GL_SHADER_STORAGE_BUFFER, vecs[1]);
		glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(int) * arraySize, vectorB.data(), GL_DYNAMIC_DRAW);
		GL_ERROR();

		glBindBuffer(GL_SHADER_STORAGE_BUFFER, vecs[2]);
		glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(int) * arraySize, nullptr, GL_DYNAMIC_DRAW);
		GL_ERROR();

		glDispatchCompute(arraySize / 512, 1, 1);
		GL_ERROR();
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

	std::cout << " mode activated.\n";
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