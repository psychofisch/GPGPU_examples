#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup(){
	std::cout << "setting up...\n";
	mLastFrame = 0.1f;

	ofBackground(69, 69, 69);

	mTestBox.setResolution(1);
	mTestBox.setScale(0.5f);
	mTestBox.setPosition(ofVec3f(0.f));

	//mMainCamera.setDistance(-100);
	//mMainCamera.setPosition(0, 0, 0);
	//mMainCamera.setupPerspective(true, 90, 0.0f, 100.f);
	mMainCamera.setPosition(0, 0, -100);
	mMainCamera.lookAt(ofVec3f(0.f));

	mLight.setPointLight();
	mLight.setPosition(ofVec3f(0.f));

	ofBoxPrimitive testRect;

	mParticleSystem = new ParticleSystem(1000);
	mParticleSystem->setDimensions(ofVec3f(50.f));
	mParticleSystem->addDamBreak(200);
	//mParticleSystem->addCube(ofVec3f(0), mParticleSystem->getDimensions(), 200);
	//mParticleSystem->addRandom();

	//mParticlesVBO.setVertexData(mParticleSystem->getPositionPtr(), 3, 1000, GL_DYNAMIC_DRAW);
	mParticlesVBO.setVertexData(mParticleSystem->getPositionPtr(), mParticleSystem->getCapacity(), GL_DYNAMIC_DRAW);
	//mParticleMesh.addVertices(mParticleSystem->getPositionPtr(), mParticleSystem->getNumberOfParticles());

	mValve = false;

	mRotationAxis = 0b000;

	mMouse = ofVec2f(-1, -1);
	mMouseSens = 0.8f;

	mHud.setup();
	mHud.add(mHudFps.setup("FPS", "XXX"));
	mHud.add(mHudRotation.setup("Rotation", "XXX"));
	mHud.add(mHudColor.setup("color", ofColor(100, 100, 140), ofColor(0, 0), ofColor(255, 255)));
	mHud.loadFromFile("settings.xml");
}

//--------------------------------------------------------------
void ofApp::update(){
	float deltaTime =  ofGetLastFrameTime();
	//std::cout << deltaTime << std::endl;

	mHudFps = ofToString(1 / deltaTime,0) + "\t" + ofToString(mParticleSystem->getNumberOfParticles()) + "/" + ofToString(mParticleSystem->getCapacity());

	float spinX = sin(ofGetElapsedTimef()*.35f);
	float spinY = cos(ofGetElapsedTimef()*.075f);

	if (mValve)
	{
		ofVec3f tmpSize = mParticleSystem->getDimensions() * 0.5f;
		mParticleSystem->addCube(tmpSize, tmpSize * ofVec3f(0.5f, 1.f, 0.5f), 1);
	}
	mParticleSystem->update(deltaTime);
	//mParticleMesh.haveVertsChanged();
	mParticlesVBO.updateVertexData(mParticleSystem->getPositionPtr(), mParticleSystem->getNumberOfParticles());
	//mTestBox.rotate(spinY, 0, 1, 0);
}

//--------------------------------------------------------------
void ofApp::draw(){
	ofEnableDepthTest();

	mMainCamera.lookAt(ofVec3f(0.f));
	//ofEnableLighting();
	//mLight.enable();
	mMainCamera.begin();

	//mTestBox.draw(ofPolyRenderMode::OF_MESH_WIREFRAME);
	
	ofPushMatrix();
	ofVec3f axis;
	float angle;
	mGlobalRotation.getRotate(angle, axis);
	ofRotate(angle, axis.x, axis.y, axis.z);
	ofTranslate(-mParticleSystem->getDimensions() * 0.5f);
	ofPushStyle();
	ofSetColor(mHudColor);
	glPointSize(5.f);
	mTestBox.drawAxes(20.f);
	//mParticleMesh.drawVertices();
	mParticlesVBO.draw(GL_POINTS, 0, mParticleSystem->getNumberOfParticles());
	ofPopStyle();
	ofPopMatrix();

	mMainCamera.end();
	//mLight.disable();
	//ofDisableLighting();

	ofDisableDepthTest();

	mHud.draw();
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){
	switch (key)
	{
	case 'v':
		mValve = true;
		break;
	default:
		break;
	}
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
		case 'r':
			mParticleSystem->setNumberOfParticles(0);
			break;
		case 'n':
			mGlobalRotation.normalize();
			mParticleSystem->setRotation(mGlobalRotation);
			mHudRotation = ofToString(mGlobalRotation);
			break;
		case 't':
			std::cout << mParticleSystem->debug_testIfParticlesOutside() << "\n";
			break; 
		case 'u':
			mParticleSystem->update(0.16f);
			break;
		case 'd':
		{
			ofVec3f tmpSize = mParticleSystem->getDimensions() * 0.5f;
			mParticleSystem->addCube(tmpSize, tmpSize * ofVec3f(0.5f, 1.f, 0.5f), 200);
		}
			break;
		case 'v':
			mValve = false;
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
	const float sens = 1.f;

	if (mRotationAxis > 0 && mMouse.x > 0.f)
	{
		//std::cout << "rotate around\t";
		if (mRotationAxis & 0b100)
		{
			mMainCamera.setPosition(mMainCamera.getPosition() + ofVec3f(0, sens * mMouseSens * (y - mMouse.y), 0));
			//std::cout << "X ";
		}

		if (mRotationAxis & 0b010)
		{
			//ofQuaternion xRotation(sens * mMouseSens * (y - mMouse.y), ofVec3f(-1, 0, 0));
			ofQuaternion yRotation(sens * mMouseSens * (x - mMouse.x), ofVec3f(0, 1, 0));
			mGlobalRotation *= yRotation;
			
			//mParticleSystem->setRotation(mGlobalRotation);

			mHudRotation = ofToString(mGlobalRotation);
			//std::cout << "Y ";
		}

		if (mRotationAxis & 0b001)
		{

			//std::cout << "Z ";
		}
		//std::cout << "axis.\n";
	}

	mMouse.x = x;
	mMouse.y = y;
}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){
	switch (button)
	{
	case OF_MOUSE_BUTTON_LEFT:
		mRotationAxis = mRotationAxis | 0b010;
		break;
	case OF_MOUSE_BUTTON_RIGHT: 
		mRotationAxis = mRotationAxis | 0b100;
		break;
	case OF_MOUSE_BUTTON_MIDDLE:
		mRotationAxis = mRotationAxis | 0b001;
		break;
	default:
		break;
	}

	mMouse.x = -1.f;
}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){
	switch (button)
	{
	case OF_MOUSE_BUTTON_LEFT:
		mRotationAxis = mRotationAxis & 0b101;
		break;
	case OF_MOUSE_BUTTON_RIGHT:
		mRotationAxis = mRotationAxis & 0b011;
		break;
	case OF_MOUSE_BUTTON_MIDDLE:
		mRotationAxis = mRotationAxis & 0b110;
		break;
	default:
		break;
	}
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
	delete mParticleSystem;

	std::cout << "quitting...\n";
	this->exit();
}
