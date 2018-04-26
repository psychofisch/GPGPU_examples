#version 440

in vec4  position;
in vec2  texcoord;
in vec4  color_coord;
in vec3  normal;

// layout(location = 0) in vec4  position;
// layout(location = 1) in vec2  texcoord;
// layout(location = 2) in vec4  color_coord;
// layout(location = 3) in vec3  normal;

layout (std140, binding = 5) buffer PositionBuffer{
	vec4 positionBuffer[];
};

uniform mat4 projectionMatrix;
uniform mat4 modelViewMatrix;
uniform mat4 modelViewProjectionMatrix;
uniform vec3 systemPos;
uniform vec4 globalColor = vec4(1.0);
uniform mat4 scale;

out vec4 colorVarying;

void main()
{	
	// when drawing instanced geometry, we can use gl_InstanceID
	// this tells you which primitive we are currently working on
	
	// we would like to spread our primitives out evenly along the x and an y coordinates
	// we calculate an x and an y coordinate value based on the current instance ID
	
	//float instanceX = float(gl_InstanceID%(iCount) - iCount/2) / 128.0;
	//float instanceY = float(gl_InstanceID/(iCount) - iCount/2) / 128.0;
	
	//vec4 vPos = (scale * position) + vec4(systemPos, 0.0) + positionBuffer[gl_InstanceID];
	vec4 vPos = (scale * position) + positionBuffer[gl_InstanceID];
	//vPos.x += instanceX;

	colorVarying = vec4(1.0);
	//gl_Position = modelViewProjectionMatrix * scale * modelViewMatrix * vPos;
	gl_Position = modelViewProjectionMatrix * vPos;
}
