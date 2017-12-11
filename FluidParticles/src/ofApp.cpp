#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup(){
	std::cout << "setting up...\n";
	mLastFrame = 0.1f;

	ofBackground(69, 69, 69);

	mTestBox.setResolution(1);
	mTestBox.setScale(0.5f);
	mTestBox.setPosition(mTestBox.getPosition() + mTestBox.getSize() * 0.25f);

	//mMainCamera.setDistance(-100);
	//mMainCamera.setPosition(0, 0, 0);
	//mMainCamera.setupPerspective(true, 90, 0.0f, 100.f);
	mMainCamera.setPosition(50, 50, -100);
	mMainCamera.lookAt(mTestBox);

	mLight.setPointLight();
	mLight.setPosition(ofVec3f(0, 0, 0));

	ofBoxPrimitive testRect;

	mParticleSystem.setDimensions(ofVec3f(50.f));
	mParticleSystem.setNumberOfParticles(1000);
	//mParticleSystem.init3DGrid(10, 10, 10, 5.0f);
	mParticleSystem.initRandom();

	//mParticlesVBO.setVertexData(mParticleSystem.getPositionPtr(), 3, 1000, GL_DYNAMIC_DRAW);
	mParticlesVBO.setVertexData(mParticleSystem.getPositionPtr(), 1000, GL_DYNAMIC_DRAW);

	mHud.setup();
	mHud.add(mHudFps.setup("FPS", "XXX"));
	mHud.add(mColor.setup("color", ofColor(100, 100, 140), ofColor(0, 0), ofColor(255, 255)));
}

//--------------------------------------------------------------
void ofApp::update(){
	float deltaTime =  ofGetLastFrameTime();
	//std::cout << deltaTime << std::endl;

	mHudFps = ofToString(1 / deltaTime);

	float spinX = sin(ofGetElapsedTimef()*.35f);
	float spinY = cos(ofGetElapsedTimef()*.075f);

	mParticleSystem.update(deltaTime);
	mParticlesVBO.updateVertexData(mParticleSystem.getPositionPtr(), 1000);
	//mTestBox.rotate(spinY, 0, 1, 0);
}

//--------------------------------------------------------------
void ofApp::draw(){
	ofEnableDepthTest();

	ofEnableLighting();
	mLight.enable();
	mMainCamera.begin();

	//mTestBox.draw(ofPolyRenderMode::OF_MESH_WIREFRAME);
	
	ofPushStyle();
	ofSetColor(mColor);
	glPointSize(5.f);
	mParticlesVBO.draw(GL_POINTS, 0, 1000);
	ofPopStyle();

	mMainCamera.end();
	mLight.disable();
	ofDisableLighting();

	ofDisableDepthTest();

	mHud.draw();
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){

}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){
	switch (key)
	{
		case OF_KEY_ESC: //quit();
			break;
		case 'h':
			std::cout << "Camera:" << mMainCamera.getPosition() << std::endl;
			std::cout << "Box: " << mTestBox.getPosition() << std::endl;
			break;
		default: std::cout << "this key hasn't been assigned\n";
			break;
	}
}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){ 

}

void ofApp::quit()
{
	std::cout << "quitting...\n";
	this->exit();
}
