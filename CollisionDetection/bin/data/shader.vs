#version 440 core
layout(location = 0) in vec3 position;
//layout(location = 1) in vec3 velocity;
//layout(location = 2) in float lifetime;

out vData{
	mat4 p;
	mat4 v;
	vec3 position;
	float age;
	vec3 pVelocity;
} vDataOut;

uniform mat4 projection;
uniform mat4 view;
uniform float maxLifetime;

void main()
{
	//gl_PointSize = 10.f * max(0.2, (lifetime / maxLifetime));
	gl_PointSize = 5.f;
	gl_Position = projection * view * vec4(position.x, position.y, position.z, 1.0f);
	vDataOut.position = position;
	vDataOut.p = projection;
	vDataOut.v = view;
	
	vDataOut.age = 1.0 - lifetime / maxLifetime;
	vDataOut.pVelocity = velocity;
}
