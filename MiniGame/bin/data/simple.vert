#version 440

// in vec4  position;
// in vec2  texcoord;
// in vec4  color_coord;
// in vec3  normal;

// openframeworks in
layout(location = 0) in vec4  position;
layout(location = 1) in vec2  texcoord;
layout(location = 2) in vec4  color_coord;
layout(location = 3) in vec3  normal;

// openframeworks uniforms (built in)
uniform mat4 projectionMatrix;
uniform mat4 modelViewMatrix;
uniform mat4 modelViewProjectionMatrix;

// custom uniforms
uniform vec3 systemPos;

out vec4 color;

void main()
{	
	vec4 vPos = vec4(systemPos, 0.0) + position;

	color = vec4(1.0);	
	gl_Position = modelViewProjectionMatrix * vPos;
}
