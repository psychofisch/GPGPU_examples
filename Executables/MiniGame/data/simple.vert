#version 440

// openframeworks in
in vec4  position;
in vec2  texcoord;
in vec4  color_coord;
in vec3  normal;

// openframeworks uniforms (built in)
uniform mat4 projectionMatrix;
uniform mat4 modelViewMatrix;
uniform mat4 modelViewProjectionMatrix;

// custom uniforms
uniform vec3 systemPos;
uniform int world;
uniform float alpha;
uniform vec3 sunDir;
uniform vec4 objColor;

out vec3 positionF;
out vec3 normalF;
out vec4 color;

void main()
{	
	vec4 vPos = vec4(systemPos, 0.0) + position;

	positionF = vPos.xyz;
	normalF = normal;
	
	normalF *= world;
	
	color = vec4(objColor.rgb, alpha);
	gl_Position = modelViewProjectionMatrix * vPos;
}
