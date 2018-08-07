__kernel void vectorAdd(
	__global const int* A,
	__global const int* B,
	__global int* C,
	const int numberOfElements
){
	uint index = get_global_id(0);

	if(index >= numberOfElements)
		return;

	C[index] = A[index] + B[index];
}
