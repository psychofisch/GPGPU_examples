__kernel void vectorAdd(
	__global int* A,
	__global int* B,
	__global int* C,
	const size_t numberOfBoxes
){
	uint index = get_global_id(0);

	if(index >= numberOfBoxes)
		return;

	C[index] = A[index] + B[index];
}
