#version 440

in vec4  position;
in vec2  texcoord;
in vec3  normal;

// openframeworks in
// layout(location = 0) in vec4  position;
// layout(location = 1) in vec2  texcoord;
// layout(location = 2) in vec4  color_coord;
// layout(location = 3) in vec3  normal;

// custom in
layout (std140, binding = 4) buffer PositionBuffer{
	vec4 positionBuffer[];
};

// openframeworks uniforms (built in)
uniform mat4 projectionMatrix;
uniform mat4 modelViewMatrix;
uniform mat4 modelViewProjectionMatrix;

// custom uniforms
uniform vec3 systemPos;
uniform vec4 globalColor = vec4(1.0);
uniform mat4 scale;
uniform int particles;
uniform int mode;

out vec4 color;
out vec3 normalF;

void main()
{	
	int particleNumber;
	vec4 vPos;
	vec3 vCol;

	// when drawing instanced geometry, we can use gl_InstanceID
	// this tells you which primitive we are currently working on
	particleNumber = gl_InstanceID;
	//vPos = (scale * position) + vec4(systemPos, 0.0) + positionBuffer[gl_InstanceID];
	vPos = (scale * position) + vec4(positionBuffer[gl_InstanceID].xyz, 0.f);

	//float val = float(gl_InstanceID)/particles;
	//vec3 vCol = mix(vec3(1.0, 0, 0), vec3(0, 1.0, 0), val);
	
	//color = vec4(vCol, 1.0);
	//gl_Position = projectionMatrix * modelViewMatrix * vPos;
	
	float val = float(particleNumber)/particles;
	vCol = mix(vec3(1.0, 0, 0), vec3(0, 1.0, 0), val);
	color = vec4(vCol, 1.0);
	
	normalF = normal;
	
	gl_Position = modelViewProjectionMatrix * vPos;
}
