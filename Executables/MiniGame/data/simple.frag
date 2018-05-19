#version 440

in vec4 color;
in vec3 positionF;
in vec3 normalF;

// custom uniforms
uniform vec3 cameraPos;
uniform vec3 sunDir;

out vec4 fragColor;

void main(){
	float ambient = 0.1;
	float diff = max(dot(normalF, sunDir), 0.0);

	fragColor = vec4((diff + ambient) * color.xyz, color.w);
}