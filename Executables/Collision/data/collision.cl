__kernel void getCollisions(
	__global float4 *minMaxBuffer,
	__global int *collisionBuffer,
	const ulong numberOfBoxes
){
	uint index = get_global_id(0);

	if(index >= numberOfBoxes)
		return;

	float3 currentMin = minMaxBuffer[(index * 2u)].xyz; //min
	float3 currentMax = minMaxBuffer[(index * 2u) + 1u].xyz; //max
	int result = -1;

	for (int j = 0; j < numberOfBoxes; j++)
	{
		if (index == j)
			continue;
		//int cnt = 0;
		float3 otherMin = minMaxBuffer[j * 2].xyz;
		float3 otherMax = minMaxBuffer[(j * 2) + 1].xyz;

		if (((  otherMin.x < currentMax.x && otherMin.x > currentMin.x)
			|| (otherMax.x < currentMax.x && otherMax.x > currentMin.x)
			|| (otherMax.x > currentMax.x && otherMin.x < currentMin.x)
			|| (otherMax.x < currentMax.x && otherMin.x > currentMin.x))
			&&
			((  otherMin.z < currentMax.z && otherMin.z > currentMin.z)
			|| (otherMax.z < currentMax.z && otherMax.z > currentMin.z)
			|| (otherMax.z > currentMax.z && otherMin.z < currentMin.z)
			|| (otherMax.z < currentMax.z && otherMin.z > currentMin.z))
			&&	
			((	otherMin.y < currentMax.y && otherMin.y > currentMin.y)
			|| (otherMax.y < currentMax.y && otherMax.y > currentMin.y)
			|| (otherMax.y > currentMax.y && otherMin.y < currentMin.y)
			|| (otherMax.y < currentMax.y && otherMin.y > currentMin.y))
			) // TODO: optimize this
		{
			result = j;
			break;// OPT: do not delete this (30% performance loss)
		}
	}

	collisionBuffer[index] = result;
}
