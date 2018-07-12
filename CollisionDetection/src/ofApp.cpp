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
	ofBoxPrimitive tmpBox;
	tmpBox.set(1.0f);
	mTemplateCube = tmpBox.getMesh();

	mCollisionSystem.setupAll(mXmlSettings);

	mMainCamera.lookAt(ofVec3f(0.f));
	//mMainCamera.setNearClip(0.01f);
	//mMainCamera.setFarClip(50.f);
	mMainCamera.setupPerspective(true, 90.f, 0.01, 10000.f);
	mMainCamera.setPosition(0, 0.f, 0.f);

	mLight.setPointLight();
	mLight.setPosition(ofVec3f(0.f, 30.f, 0.f));

	int maxParticles = mXmlSettings.getValue("GENERAL:MAXPARTICLES", 5000);
	if (maxParticles <= 0)
	{
		std::cout << "WARNING: GENERAL:MAXPARTICLES was \"<= 0\" again.\n";
		maxParticles = 5000;
	}

	//mTargetCollisions = mXmlSettings.getValue("GENERAL:COLPERC", 0.1f);

	mLockMouse = false;

	mMouse = vec2i(-1, -1);
	mMouseSens = mXmlSettings.getValue("CONTROLS:MOUSESENS", 0.8f);

	// box shader setup
	mBoxShader.load(mXmlSettings.getValue("GENERAL:VERT", ""), mXmlSettings.getValue("GENERAL:FRAG", ""));
	HANDLE_GL_ERROR();

	// HUD setup
	mHudDebugGroup.setName("Program Information");
	mHudDebugGroup.add(mHudFps.set("FPS", -1.f));
	mHudDebugGroup.add(mHudMode.set("Mode", "XXX"));
	mHudDebugGroup.add(mHudCubes.set("Cubes", ofToString(mCubes.size())));
	mHudDebugGroup.add(mHudCollisionPercentage.set("Collisions", "??%"));
	mHudDebugGroup.add(mHudMeasureNext.set("Measure Next", false));

	mHudControlGroup.setName("Settings");
	mHudControlGroup.add(mHudMovement.set("Movement", true));
	mHudControlGroup.add(mHudDraw.set("Draw", true));
	mHudControlGroup.add(mHudCollision.set("Collisions", true));
	
	mHud.setup();
	mHud.add(mHudDebugGroup);
	mHud.add(mHudControlGroup);
	mHud.loadFromFile("hud.xml");

	mHudMode = mCollisionSystem.getModeAsString();
	mHudCubes = ofToString(mCubes.size());
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
	int colCount = 0;

	for (int i = 0; i < mCubes.size() && mHudMovement; ++i)
	{
		float r = ofRandom(0.8f, 1.2f);

		//pos.setPosition(mCubes[i].getPosition());
		pos.setPosition(mCubePosAndSize[i * 2]);

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

		mCubePosAndSize[i * 2] = pos.getPosition();

		// DEBUG: collision percentage
		if (mCollisions[i] > -1)
		{
			colCount++;
		}
		//*** DEBUG
	}

	// DEBUG: collision percentage
	if (mCollisions.size() > 0)
		mHudCollisionPercentage = ofToString(100.f * colCount / mCollisions.size(), 2);
	else
		mHudCollisionPercentage = "0";
	//*** DEBUG
	//*** mc

	HANDLE_GL_ERROR();
	if (mCubePosAndSize.size() > 0 && mPosAndSize.size() < sizeof(ofVec4f) * mCubePosAndSize.size())
	{
		std::cout << "mCubePosAndSize reallocate\n";
		mPosAndSize.allocate(mCubePosAndSize, GL_DYNAMIC_DRAW);
		HANDLE_GL_ERROR();
	}
	else
		mPosAndSize.updateData(mCubePosAndSize);
	HANDLE_GL_ERROR();
	//*** mc

	// Collision detection
	if (mHudCollision)
	{
		if (mHudMeasureNext)
			mCollisionSystem.measureNextCalculation();

		mCollisionSystem.getCollisions(mCubes, mCollisions);

		if (mCollisions.size() > 0)
		{
			if (mGPUCollisions.size() < sizeof(int) * mCollisions.size())
			{
				std::cout << "mGPUCollisions reallocate\n";
				mGPUCollisions.allocate(mCollisions, GL_DYNAMIC_DRAW);
				//mPosAndSize.allocate(sizeof(ofVec4f) * mCubePosAndSize.size(), GL_DYNAMIC_DRAW);
			}else
				mGPUCollisions.updateData(mCollisions);
		}

		if (mHudMeasureNext)
		{
			std::cout << mCollisionSystem.getMeasurement() << std::endl;
			mHudMeasureNext = false;
		}
	}
	//*** cd

	mHudCubes = ofToString(mCubes.size());
}

//--------------------------------------------------------------
void ofApp::draw(){
	ofEnableDepthTest();

	mMainCamera.begin();

	if (mCubes.size() > 0)
	{
		mBoxShader.begin();

		mPosAndSize.bindBase(GL_SHADER_STORAGE_BUFFER, 0);
		mGPUCollisions.bindBase(GL_SHADER_STORAGE_BUFFER, 1);

		mBoxShader.setUniform1i("noOfBoxes", mCubes.size());
		mBoxShader.setUniform1i("collisionsOn", (mHudCollision) ? 1 : 0);
		mBoxShader.setUniform3f("sunPos", ofVec3f(0, 0, 0));

		mTemplateCube.drawInstanced(ofPolyRenderMode::OF_MESH_FILL, mCubes.size());

		mBoxShader.end();
	}

	mMainCamera.end();

	ofDisableDepthTest();

	mHud.draw();

	HANDLE_GL_ERROR();
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
			/*mXmlSettings.loadFile("settings.xml");
			mTargetCollisions = mXmlSettings.getValue("GENERAL:COLPERC", 0.1f);
			resetCubes(mCubes.size());*/
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
			mHudMode = mCollisionSystem.getModeAsString();
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
	float baseBoxSize = 14.95f * powf(numberOfCubes, -0.36f);// this scales the boxes so that ~15-20% of the boxes collide at all times

	Cube templateCube;
	templateCube.setResolution(1);
	templateCube.setScale(1.f);
	templateCube.set(baseBoxSize);
	templateCube.setPosition(ofVec3f(0.f));
	templateCube.disableColors();
	templateCube.disableTextures();

	int boxNumber = numberOfCubes;
	float side = templateCube.getWidth();
	ofVec3f boxPos(0.f);
	ofVec3f direction;
	float ringSize = 30.f;//fixed ring size
	float ringWidth = 0.3f;//relative to ring size
	mCubes.resize(boxNumber, templateCube);
	mCollisions.resize(boxNumber, -1);
	mCubePosAndSize.resize(boxNumber * 2);
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
		mCubePosAndSize[(i * 2) + 1] = mCubes[i].getSize();

		//position
		//mCubes[i].setPosition(boxPos);
		mCubePosAndSize[i * 2] = boxPos;

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
	mXmlSettings.setValue("CONTROLS:MOUSESENS", mMouseSens);
	mXmlSettings.saveFile("settings.xml");
	std::cout << "done\n";

	std::cout << "saving HUD...";
	mHud.saveToFile("hud.xml");
	std::cout << "done\n";

	std::cout << "quitting...bye =)\n";
	this->exit();
}
