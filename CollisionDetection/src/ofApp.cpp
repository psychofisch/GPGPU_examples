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

	resetCubes(mXmlSettings.getValue("GENERAL:BOXES", 1000));

	mCollisionSystem.setupAll(mXmlSettings);

	mMainCamera.lookAt(ofVec3f(0.f));
	//mMainCamera.setNearClip(0.01f);
	//mMainCamera.setFarClip(50.f);
	mMainCamera.setupPerspective(true, 90.f, 0.01, 10000.f);
	mMainCamera.setPosition(0, 10.f, 10.f);

	mLight.setPointLight();
	mLight.setPosition(ofVec3f(0.f, 30.f, 0.f));

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
	mHudControlGroup.add(mHudCubes.set("Cubes", ofToString(mCubes.size())));
	mHudControlGroup.add(mHudMovement.set("Movement", true));
	mHudControlGroup.add(mHudDraw.set("Draw", true));
	mHudControlGroup.add(mHudCollision.set("Collisions", true));
	//mHudControlGroup.add(mHudWorkGroup.set("Workgroup Size", tmpCUDA.maxWorkGroupSize, 1, tmpCUDA.maxWorkGroupSize));

	mHud.setup();
	mHud.add(mHudDebugGroup);
	mHud.add(mHudControlGroup);
	mHud.loadFromFile("hud.xml");

	mHudMode = iHudGetModeString(mCollisionSystem.getMode());
}

//--------------------------------------------------------------
void ofApp::update(){
	float dt =  std::min(0.1, ofGetLastFrameTime());

	mHudFps = std::round(ofGetFrameRate());

	ofVec3f moveVec;
	moveVec.x = mMoveVec.x * 0.1f;
	moveVec.z = mMoveVec.y * 0.1f;
	moveVec.y = 0.f;

	moveVec *= 500.f * std::max(mCubes.size() * 0.001f, 1.f) * dt;

	mMainCamera.dolly(-moveVec.z);
	mMainCamera.truck(moveVec.x);

	// move cubes
	ofSeedRandom(1337);//seed every frame so that every cube has a constant "random" speed value
	ofNode pos;
	for (size_t i = 0; i < mCubes.size() && mHudMovement; ++i)
	{
		float r = ofRandom(0.8f, 1.2f);

		pos.setPosition(mCubes[i].getPosition());

		ofVec3f axis = vec3::left;
		if (i > mCubes.size() * 0.666f)
		{
			axis = vec3::up;
		}
		else if (i > mCubes.size() * 0.333f)
		{
			axis = vec3::forward;
		}

		pos.rotateAround(30.f * r * dt, axis, ofVec3f::zero());
		
		mCubes[i].setPosition(pos.getPosition());
	}
	//*** mc

	// Collision detection
	if (mHudCollision)
	{
		mCollisionSystem.getCollisions(mCubes, mCollisions);
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

	for (int i = 0; i < mCubes.size() && mHudDraw; ++i)
	{
		if(mCollisions[i] >= 0)
			ofSetColor(ofColor::red);
		else
			ofSetColor(mCubes[i].mColor);
		mCubes[i].draw();
		//mCubes[i].drawWireframe();
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
			break;
		case 'r':
			mCubes.resize(0);
			mCollisions.resize(0);
			mHudCubes = ofToString(mCubes.size());
			break;
		case 'e':
			resetCubes(mCubes.size() + mXmlSettings.getValue("GENERAL:ADD", 100));
			mHudCubes = ofToString(mCubes.size());
			break;
		case 't':
			break; 
		case 'u':
			break;
		case 'c':
			mLockMouse = !mLockMouse;
			/*{
				ofVec3f tmpSize = mParticleSystem->getDimensions() * 0.5f;
				mParticleSystem->addCube(ofVec3f(0, tmpSize.y, 0), tmpSize * ofVec3f(0.5f, 1.f, 0.5f), 500, true);
			}*/
			break;
		case 'm':
		{
			CollisionSystem::ComputeMode currentMode = mCollisionSystem.nextMode(mCollisionSystem.getMode());
			mCollisionSystem.setMode(currentMode);
			mHudMode = iHudGetModeString(currentMode);
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

void ofApp::resetCubes(int numberOfCubes)
{
	Cube templateCube;
	templateCube.setResolution(1);
	templateCube.setScale(1.f);
	templateCube.set(10.f);
	templateCube.setPosition(ofVec3f(0.f));

	int boxNumber = numberOfCubes;
	int sqrtBox = sqrtf(boxNumber);
	float side = templateCube.getWidth();
	ofVec3f boxPos(0.f);
	ofVec3f direction;
	float ringSize = std::max(boxNumber * 0.15f, side * 5.f);
	float ringWidth = 0.3f;
	mCubes.resize(boxNumber, templateCube);
	mCollisions.resize(boxNumber, false);
	for (int i = 0; i < boxNumber; ++i)
	{
		if (i > boxNumber * 0.666f)
		{
			direction.x = ofRandom(-1.f, 1.0f);
			direction.y = ofRandom(-0.1f, 0.1f);
			direction.z = ofRandom(-1.f, 1.0f);
		}
		else if (i > boxNumber * 0.333f)
		{
			direction.x = ofRandom(-1.f, 1.0f);
			direction.z = ofRandom(-0.1f, 0.1f);
			direction.y = ofRandom(-1.f, 1.0f);
		}
		else
		{
			direction.z = ofRandom(-1.f, 1.0f);
			direction.x = ofRandom(-0.1f, 0.1f);
			direction.y = ofRandom(-1.f, 1.0f);
		}
		direction.normalize();

		boxPos = direction * (ringSize * ofRandom(1.f - ringWidth, 1.f + ringWidth));

		//shape
		mCubes[i].set(side * ofRandom(0.5f, 1.5f), side * ofRandom(0.5f, 1.5f), side * ofRandom(0.5f, 1.5f));

		//position
		mCubes[i].setPosition(boxPos);

		//color
		mCubes[i].mColor = ofColor::cyan;
		/*if (i % 2 == 0)
			mCubes[i].mColor = ofColor::cyan;
		else
			mCubes[i].mColor = ofColor::magenta;*/

		//minMax
		mCubes[i].recalculateMinMax();
	}
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

std::string ofApp::iHudGetModeString(CollisionSystem::ComputeMode m)
{
	if (m == CollisionSystem::ComputeMode::CPU)
		return "CPU";
	else if (m == CollisionSystem::ComputeMode::COMPUTE_SHADER)
		return "Compute Shader";
	else if (m == CollisionSystem::ComputeMode::OPENCL)
		return "OpenCL";
	else if (m == CollisionSystem::ComputeMode::CUDA)
		return "CUDA";
	else if (m == CollisionSystem::ComputeMode::THRUST)
		return "Thrust";
	else
		return "UNKNOWN";
}
