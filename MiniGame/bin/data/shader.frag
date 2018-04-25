#version 440

in vec4 colorVarying;

out vec4 fragColor;

void main(){
	fragColor = colorVarying;
}