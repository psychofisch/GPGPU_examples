#version 440

in vec4 color;
in vec3 normalF;

// custom uniforms
uniform vec3 cameraPos;
uniform vec3 sunPos;

out vec4 fragColor;

void main(){
	float ambient = 0.1;
	float diff = 1.0;
	//float diff = max(dot(normalF, sunDir), 0.0);

	fragColor = vec4((diff + ambient) * color);
}