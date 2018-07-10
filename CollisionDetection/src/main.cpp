#include "ofMain.h"
#include "ofApp.h"

//========================================================================
int main(int argc, const char * argv){
	ofGLFWWindowSettings settings;
	settings.setGLVersion(4, 6); //version of opengl corresponding to your GLSL version
	settings.width = 1024;
	settings.height = 768;
	ofCreateWindow(settings);

	//ofSetupOpenGL(1024,768,OF_WINDOW);			// <-------- setup the GL context

	ofRunApp(new ofApp());
}
