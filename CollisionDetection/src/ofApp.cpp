#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup(){
	std::cout << "setting up...\n";

	if (mXmlSettings.loadFile("settings.xml")) {
		std::cout << "settings.xml loaded\n";
	}
	else {
		std::cout << "unable to load settings.xml check data/ folder\n";
		std::cout << "creating default settings.xml\n";
		mXmlSettings.load("settings.xml.bk");
		mXmlSettings.save("settings.xml");
		mXmlSettings.loadFile("settings.xml");
	}

	ofSetVerticalSync(false);

	ofBackground(69, 69, 69);

	mTestCube.setResolution(1);
	mTestCube.setScale(1.f);
	mTestCube.set(10.f);
	mTestCube.setPosition(ofVec3f(0.f));

	int boxNumber = 1000;
	int sqrtBox = sqrtf(boxNumber);
	float gap = 10.f;
	float side = mTestCube.getWidth();
	float offset = -(side + gap) * sqrtBox * 0.5f;
	ofVec3f boxPos(0.f);
	boxPos.x = boxPos.z = offset;
	mBoxes.resize(boxNumber, mTestCube);
	mCollisions.resize(boxNumber, false);
	for (int i = 0; i < boxNumber; ++i)
	{
		boxPos.x += gap + side;
		if (i % sqrtBox == 0)
		{
			boxPos.x = offset;
			boxPos.z += gap + side;
		}

		mBoxes[i].set(side * ofRandom(0.7f, 1.3f), side * ofRandom(0.7f, 1.3f), side * ofRandom(0.7f, 1.3f));

		mBoxes[i].setPosition(boxPos);
		if (i % 2 == 0)
			mBoxes[i].mColor = ofColor::cyan;
		else
			mBoxes[i].mColor = ofColor::magenta;
	}

	mMainCamera.lookAt(ofVec3f(0.f));
	//mMainCamera.setNearClip(0.01f);
	//mMainCamera.setFarClip(50.f);
	mMainCamera.setupPerspective(true, 90.f, 0.01, 1000.f);
	mMainCamera.setPosition(0, 10.f, 10.f);

	mLight.setPointLight();
	mLight.setPosition(ofVec3f(0.f, 30.f, 0.f));

	ofBoxPrimitive testRect;

	int maxParticles = mXmlSettings.getValue("GENERAL:MAXPARTICLES", 5000);
	if (maxParticles <= 0)
	{
		std::cout << "WARNING: GENERAL:MAXPARTICLES was \"<= 0\" again.\n";
		maxParticles = 5000;
	}

	mLockMouse = false;

	mMouse = vec2i(-1, -1);
	mMouseSens = mXmlSettings.getValue("CONTROLS:MOUSESENS", 0.8f);

	mHudDebugGroup.setName("Debug Information");
	mHudDebugGroup.add(mHudFps.set("FPS", -1.f));
	//mHudDebugGroup.add(mHudRotation.set("Rotation", mGlobalRotation));

	mHudControlGroup.setName("Program Information");
	mHudControlGroup.add(mHudMode.set("Mode", "XXX"));
	//mHudControlGroup.add(mHudWorkGroup.set("Workgroup Size", tmpCUDA.maxWorkGroupSize, 1, tmpCUDA.maxWorkGroupSize));
	mHudControlGroup.add(mHudParticles.set("Particles", "0/XXX"));
	mHudControlGroup.add(mHudColor.set("Particle Color", ofColor(100, 100, 140)));

	mHud.setup();
	mHud.add(mHudDebugGroup);
	mHud.add(mHudControlGroup);
	mHud.loadFromFile("hud.xml");
}

//--------------------------------------------------------------
void ofApp::update(){
	float dt =  std::min(0.1, ofGetLastFrameTime());

	mHudFps = std::round(ofGetFrameRate());

	ofVec3f moveVec;
	moveVec.x = mMoveVec.x * 0.1f;
	moveVec.z = mMoveVec.y * 0.1f;
	moveVec.y = 0.f;

	moveVec *= 500.f * dt;

	mMainCamera.dolly(-moveVec.z);
	mMainCamera.truck(moveVec.x);

	ofSeedRandom(1337);
	for (int i = 0; i < mBoxes.size(); ++i)
	{
		float r = ofRandom(0.5f, 1.5f);
		float sign = 1.f;
		if (i % 2 == 0)
			sign *= -1.f;
		ofNode pos;
		pos.setPosition(mBoxes[i].getPosition());
		pos.rotateAround(sign * 30.f * r * dt, ofVec3f(0.f, 1.f, 0.f), ofVec3f(0.f));
		mBoxes[i].setPosition(pos.getPosition());
	}

	// Collision detection
	std::vector<ofVec3f[2]> minMax(mBoxes.size());
	for (size_t i = 0; i < mBoxes.size(); ++i) // calculate bounding boxes
	{
		const std::vector<ofVec3f>& vertices = mBoxes[i].getMesh().getVertices();
		ofVec3f min, max, pos;
		min = ofVec3f(INFINITY);
		max = ofVec3f(-INFINITY);
		pos = mBoxes[i].getPosition();
		for (size_t o = 0; o < vertices.size(); o++)
		{
			ofVec3f current = vertices[o] + pos;
			for (size_t p = 0; p < 3; p++)
			{
				if (current[p] < min[p])
					min[p] = current[p];
				else if (current[p] > max[p])
					max[p] = current[p];
			}
		}
		minMax[i][0] = min;
		minMax[i][1] = max;
	}

	for (size_t i = 0; i < minMax.size(); i++)
	{
		ofVec3f currentMin = minMax[i][0];
		ofVec3f currentMax = minMax[i][1];
		mCollisions[i] = false;
		for (size_t j = 0; j < minMax.size(); j++)
		{
			if (i == j)
				continue;
			int cnt = 0;
			for (size_t p = 0; p < 3; p++)
			{
				ofVec3f otherMin = minMax[j][0];
				ofVec3f otherMax = minMax[j][1];
				if (	(otherMin[p] < currentMax[p] && otherMin[p] > currentMin[p])
					||	(otherMax[p] < currentMax[p] && otherMax[p] > currentMin[p])
					||	(otherMax[p] > currentMax[p] && otherMin[p] < currentMin[p])
					||	(otherMax[p] < currentMax[p] && otherMin[p] > currentMin[p]))
					cnt++;
			}

			if (cnt >= 3)
			{
				mCollisions[i] = true;
				break;
			}
		}
	}
	//*** cd

	if (!mHudPause || mHudStep)
	{
		float deltaTime = dt;
		if (mHudStep)
			deltaTime = 0.008f;
		//mParticleSystem->update(dt);
		mHudStep = false;
	}
}

//--------------------------------------------------------------
void ofApp::draw(){
	ofEnableDepthTest();

	//mMainCamera.lookAt(ofVec3f(0.f));
	ofEnableLighting();
	mLight.enable();
	mMainCamera.begin();

	//mTestCube.draw(ofPolyRenderMode::OF_MESH_WIREFRAME);
	
	ofPushMatrix();
	//ofRotate(angle, axis.x, axis.y, axis.z);
	//ofTranslate(-mParticleSystem->getDimensions() * 0.5f);
	ofPushStyle();

	for (int i = 0; i < mBoxes.size(); ++i)
	{
		if(mCollisions[i])
			ofSetColor(ofColor::red);
		else
			ofSetColor(mBoxes[i].mColor);
		mBoxes[i].draw();
		//mBoxes[i].drawWireframe();
	}
	//ofSetColor(mHudColor);
	//mTestCube.draw();

	ofPopStyle();
	ofPopMatrix();

	mMainCamera.end();
	mLight.disable();
	ofDisableLighting();

	ofDisableDepthTest();

	mHud.draw();
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){
	switch (key)
	{
	case OF_KEY_ESC: quit();
		break;
	case 'v':
		break;
	case 'w':
		mMoveVec.y = 1;
		break;
	case 's':
		mMoveVec.y = -1;
		break;
	case 'd':
		mMoveVec.x = 1;
		break;
	case 'a':
		mMoveVec.x = -1;
		break;
	default:
		break;
	}
}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){
	switch (key)
	{
		case 'w'://fallthrough
		case 's':
			mMoveVec.y = 0;
		break;
		case 'd'://fallthrough
		case 'a':
			mMoveVec.x = 0;
			break;
		case 'h':
			std::cout << "Camera:" << mMainCamera.getPosition() << std::endl;
			std::cout << "Camera:" << mMainCamera.getGlobalOrientation() << std::endl;
			std::cout << "Box: " << mTestCube.getPosition() << std::endl;
			break;
		case 'r':
			break;
		case 'n':
			break;
		case 't':
			break; 
		case 'u':
			break;
		case 'v':
			mLockMouse = !mLockMouse;
			/*{
				ofVec3f tmpSize = mParticleSystem->getDimensions() * 0.5f;
				mParticleSystem->addCube(ofVec3f(0, tmpSize.y, 0), tmpSize * ofVec3f(0.5f, 1.f, 0.5f), 500, true);
			}*/
			break;
		case 'm':
		{
		}
			break;
		default: std::cout << "this key hasn't been assigned\n";
			break;
	}
}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){

	if (mLockMouse)
		return;

	int distX = (x - mMouse.x);
	int distY = (y - mMouse.y);

	mCameraRotation.y += mMouseSens * distX;
	mCameraRotation.x += mMouseSens * distY;

	mMainCamera.setOrientation(mCameraRotation);

	mMouse.x = x;
	mMouse.y = y;
}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){
}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){
	switch (button)
	{
	case OF_MOUSE_BUTTON_LEFT:
		break;
	case OF_MOUSE_BUTTON_RIGHT: 
		break;
	case OF_MOUSE_BUTTON_MIDDLE:
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
		break;
	case OF_MOUSE_BUTTON_RIGHT:
		break;
	case OF_MOUSE_BUTTON_MIDDLE:
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
	std::cout << "saving settings...";
	/*uint tmpCapacity = mParticleSystem->getCapacity();
	if (tmpCapacity > INT_MAX)
	{
		tmpCapacity = INT_MAX;
	}
	mXmlSettings.setValue("GENERAL:MAXPARTICLES", static_cast<int>(tmpCapacity));//BUG: this doesn't work
	//mXmlSettings.setValue("GENERAL:MAXPARTICLES", static_cast<int>(100000u));//^BUG: this does work?!?!*/
	mXmlSettings.setValue("CONTROLS:MOUSESENS", mMouseSens);
	//mXmlSettings.setValue("SIM:SWIDTH", mHud);
	mXmlSettings.saveFile("settings.xml");

	mHud.saveToFile("hud.xml");
	std::cout << "done\n";

	std::cout << "quitting...bye =)\n";
	this->exit();
}

std::string ofApp::iHudGetModeString(ParticleSystem::ComputeMode m)
{
	if (m == ParticleSystem::ComputeMode::CPU)
		return "CPU";
	else if (m == ParticleSystem::ComputeMode::COMPUTE_SHADER)
		return "Compute Shader";
	else if (m == ParticleSystem::ComputeMode::OPENCL)
		return "OpenCL";
	else if (m == ParticleSystem::ComputeMode::CUDA)
		return "CUDA";
	else if (m == ParticleSystem::ComputeMode::THRUST)
		return "Thrust";
	else
		return "UNKNOWN";
}
