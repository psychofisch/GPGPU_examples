#version 440

// openframeworks in
in vec4  position;
//in vec2  texcoord;
//in vec4  color_coord;
in vec3  normal;

//custom in
layout (std140, binding = 0) buffer PositionBuffer{
	vec4 positionBuffer[];
};

layout (packed, binding = 1) buffer CollisionBuffer{
	int collisionBuffer[];
};

// openframeworks uniforms (built in)
uniform mat4 projectionMatrix;
uniform mat4 modelViewMatrix;
uniform mat4 modelViewProjectionMatrix;

// custom uniforms
uniform int noOfBoxes;
uniform vec3 sunPos;
uniform vec4 objColor;
uniform int collisionsOn;

out vec3 normalF;
out vec4 color;

void main()
{	
	//vec4 vPos = (position * vec4(2.0, 1.0, 1.0, 1.0)) + positionBuffer[gl_InstanceID * 2];
	vec4 vPos = (position * vec4(positionBuffer[(gl_InstanceID * 2) + 1].xyz, 1.0)) + positionBuffer[gl_InstanceID * 2];
	
	normalF = normal;
	
	color = vec4(0.0, 1.0, 1.0, 1.0);
	
	if(collisionsOn > 0 && collisionBuffer[gl_InstanceID] > -1)
		color = vec4(1.0, 0.0, 0.0, 1.0);
	
	gl_Position = modelViewProjectionMatrix * vPos;
}
