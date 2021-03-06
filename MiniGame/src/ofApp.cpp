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

	// graphic setup
	ofGetWindowPtr()->setWindowTitle(mXmlSettings.getValue("GENERAL:TITLE", "Minigame"));

	ofSetFrameRate(144);
	ofSetVerticalSync(false);

	ofBackground(69, 69, 69);

	mMainCamera.setPosition(1, 2, 1);
	mMainCamera.lookAt(ofVec3f(0.f));
	mMainCamera.setNearClip(0.01f);
	mMainCamera.setFarClip(50.f);
	mMainCamera.setFov(50.f);

	mWorld.set(1.0f);
	mWorld.setPosition(ofVec3f(0.5f));

	// particle setup
	int maxParticles = mXmlSettings.getValue("GENERAL:MAXPARTICLES", 5000);
	if (maxParticles <= 0)
	{
		std::cout << "WARNING: GENERAL:MAXPARTICLES was \"<= 0\" again.\n";
		maxParticles = 5000;
	}

	mParticleSystem = new ParticleSystem(maxParticles);
	mParticleSystem->setupAll(mXmlSettings);
	mParticleSystem->setMode(ParticleSystem::ComputeMode::CPU);
	mParticleSystem->setDimensions(ofVec3f(1.f, 1.f, 1.f));
	mParticleSystem->createParticleShader(mXmlSettings.getValue("GENERAL:VERT", "shader.vert"), mXmlSettings.getValue("GENERAL:FRAG", "shader.frag"));
	//mParticleSystem->addDamBreak(200);
	//mParticleSystem->addCube(ofVec3f(0), mParticleSystem->getDimensions(), 200);
	//mParticleSystem->addRandom();

	mValve = false;

	mMoveVec.y = 1.f;

	// control and HUD setup
	mRotationAxis = 0b000;

	mMouse = ofVec2f(-1, -1);
	mMouseSens = mXmlSettings.getValue("CONTROLS:MOUSESENS", 0.8f);

	mHudDebugGroup.setName("Debug Information");
	mHudDebugGroup.add(mHudFps.set("FPS", -1.f));
	//mHudDebugGroup.add(mHudRotation.set("Rotation", mGlobalRotation));

	mHudControlGroup.setName("Program Information");
	mHudControlGroup.add(mHudMode.set("Mode", "XXX"));
	//mHudControlGroup.add(mHudWorkGroup.set("Workgroup Size", tmpCUDA.maxWorkGroupSize, 1, tmpCUDA.maxWorkGroupSize));
	mHudControlGroup.add(mHudParticles.set("Particles", "0/XXX"));
	mHudControlGroup.add(mHudTime.set("Time", 1.0f, 0.f, 5.f));

	mHudSimulationGroup.setName("Simulation Settings");
	mHudSimulationGroup.add(mHudPause.set("Pause", false));
	mHudSimulationGroup.add(mHudStep.set("Step", false));
	mHudSimulationGroup.add(mHudInteractionRadius.set("Interaction Radius", 0.1f, 0.00000001f, 1.f));
	mHudSimulationGroup.add(mHudPressureMultiplier.set("Pressure Multiplier", 0.5f, 0.0f, 10.f));
	mHudSimulationGroup.add(mHudViscosity.set("Viscosity", 1.0f, 0.0f, 1000.f));
	mHudSimulationGroup.add(mHudRestPressure.set("Rest Pressure", .1f, 0.0f, 1.f));

	mHudGameGroup.setName("Game Info");
	mHudGameGroup.add(mHudGameGameState.set("Gamestate", "???"));
	mHudGameGroup.add(mHudGameTime.set("Time", -1.f));
	mHudGameGroup.add(mHudGameHelp.set("Help", HELP_START));

	mHud.setup();
	mHud.add(mHudDebugGroup);
	mHud.add(mHudControlGroup);
	mHud.add(mHudSimulationGroup);
	mHud.add(mHudGameGroup);
	mHud.loadFromFile("hud.xml");

	mHudMode = iHudGetModeString(mParticleSystem->getMode());

	std::cout << "OpenGL " << ofGetGLRenderer()->getGLVersionMajor() << "." << ofGetGLRenderer()->getGLVersionMinor() << std::endl;

	mMainFont.loadFont("Roboto-Regular.ttf", 64, true, true, true);

	// level setup
	mLevel.setParticleSystem(mParticleSystem);

	Cube tmpCube;
	tmpCube.set(.25f);

	tmpCube.set(0.1f, 0.3f, 0.8f);
	tmpCube.setPosition(ofVec3f(0.4f, -0.01f, -0.01f) + (tmpCube.getSize() * 0.5f));
	mLevel.addLevelCollider(tmpCube);

	tmpCube.setPosition(ofVec3f(0.699f, -0.01f, 0.21f) + (tmpCube.getSize() * 0.5f));
	mLevel.addLevelCollider(tmpCube);

	mLevel.setEndzone(ofVec3f(0.9f, 0.1f, 0.9f), ofVec3f(0.2f));

	mLevel.setStartzone(ofVec3f(0.f, 0.8f, 0.f), ofVec3f(0.2f), mXmlSettings.getValue("GENERAL:DROPSIZE", 100));

	mLevelShader.load("simple.vert", "simple.frag");
	mLevel.setLevelShader(&mLevelShader);

	mSunDirection = ofVec3f(1.0f, 2.0f, 0.f).getNormalized();
	mLevel.setSunDirection(&mSunDirection);

	mLevel.setTimeToFinish(5.f);

	mLevel.setReady(true);
	// *** ls
}

//--------------------------------------------------------------
void ofApp::update(){
	float deltaTime =  std::min(0.1, ofGetLastFrameTime());
	//std::cout << deltaTime << std::endl;

	mHudFps = std::round(ofGetFrameRate());
	mHudParticles = ofToString(mParticleSystem->getNumberOfParticles()) + "/" + ofToString(mParticleSystem->getCapacity());
	//mHudFps = ofToString(ofGetFrameRate(),0) + "\t" + ofToString(mParticleSystem->getNumberOfParticles()) + "/" + ofToString(mParticleSystem->getCapacity());

	SimulationData simD;
	simD.interactionRadius = mHudInteractionRadius;
	simD.pressureMultiplier = mHudPressureMultiplier;
	simD.viscosity = mHudViscosity;
	simD.restPressure = mHudRestPressure;
	mParticleSystem->setSimulationData(simD);
	//mParticleSystem->setSmoothingWidth(mHudInteractionRadius);
	//mParticleSystem->setRestDensity(mHudPressureMultiplier);
	//mParticleSystem->getCudata().maxWorkGroupSize = mHudWorkGroup;

	float spinX = sin(ofGetElapsedTimef()*.35f);
	float spinY = cos(ofGetElapsedTimef()*.075f);

	if (mValve)
	{
		ofVec3f tmpSize = mParticleSystem->getDimensions();
		mParticleSystem->addCube(tmpSize * ofVec3f(0.5f, 1.0f, 0.5f), tmpSize * 0.1f, 2, true);
	}

	//ofVec3f moveVec = mMoveVec /** deltaTime*/;
	//mParticleSystem->addPosition(moveVec);
	//mTestBox.move(moveVec);
	ofVec3f gravity = Particle::gravity.y * mMoveVec.normalized();
	mParticleSystem->setGravity(gravity);
	 
	Level::GameState prevState = mLevel.getGameState();

	if (!mHudPause || mHudStep)
	{
		float dt = deltaTime;
		dt *= mHudTime;

		if (mHudStep)
			dt = 0.033f;

		// collision preparation
		mLevel.update(dt);

		// update particles
		mParticleSystem->update(dt);

		// reset this bool if step mode is activated (no unnecessary branching)
		mHudStep = false;
	}
	//mParticleSystem->update(0.016f);
	//mParticleMesh.haveVertsChanged();
	//mTestBox.rotate(spinY, 0, 1, 0);

	// HUD stuff
	Level::GameState state = mLevel.getGameState();
	
	if (prevState != state)
	{
		mHudGameGameState = Level::convertGamestateToString(state);
	}

	if (state == Level::GameState::Running)
	{
		float max = mLevel.getLevelTime();
		mHudGameTime.setMax(max);
		mHudGameTime = max - mLevel.getCurrentTime();
	}
	else if (state == Level::GameState::Ready)
	{
		mHudGameHelp = HELP_START;
	}

	if (prevState == Level::GameState::Running)
	{
		if (state == Level::GameState::Finished || state == Level::GameState::TimeOver)
		{
			std::string pre;
			if (state == Level::GameState::TimeOver)
			{
				pre = "TIMES UP";
			}
			else
				pre = "FINISHED";

			uint score = mLevel.getScore();
			uint particles = mLevel.getSpawnedParticles();

			mTextDuration = HUGE_VALF;
			mMainString = pre + "\nScore = " + ofToString(score) + "/" + ofToString(particles);
		}
	}
	else if(prevState == Level::GameState::Ready)
	{
		mTextDuration = 0.f;
	}

	if (mTextDuration > 0.f)
		mTextDuration -= deltaTime;
	if(mTextDuration < 0.f)
		mTextDuration = 0.f;
}

//--------------------------------------------------------------
void ofApp::draw(){
	ofEnableDepthTest();

	mMainCamera.lookAt(ofVec3f(0));
	mMainCamera.begin();
	
	// prerequisites
	ofVec3f axis;
	float angle;
	mGlobalRotation.getRotate(angle, axis);
	ofRotate(angle, axis.x, axis.y, axis.z);

	ofTranslate(-mParticleSystem->getDimensions() * ofVec3f(0.5f, 0.25f, 0.5f));
	// *** p

	// draw game
	mParticleSystem->draw(mMainCamera.getPosition(), mSunDirection, ofPolyRenderMode::OF_MESH_FILL);

	mLevel.draw(mMainCamera.getPosition(), ofPolyRenderMode::OF_MESH_FILL);

	glEnable(GL_CULL_FACE);
	glCullFace(GL_FRONT);
	mLevelShader.begin();
	mLevelShader.setUniform3f("systemPos", ofVec3f(0.f));
	mLevelShader.setUniform1i("endZone", 0);
	mLevelShader.setUniform3f("cameraPos", mMainCamera.getPosition());
	mLevelShader.setUniform3f("sunDir", mSunDirection);
	mLevelShader.setUniform1i("world", -1);
	mLevelShader.setUniform4f("objColor", ofFloatColor::wheat);
	mLevelShader.setUniform1f("alpha", 1.f);

	mWorld.draw();

	mLevelShader.end();
	glDisable(GL_CULL_FACE);

	ofDisableDepthTest();

	ofFill();
	ofVec3f pos(0.5f, 0.7f, 0.5f);
	float arrowLength = 0.2f;
	ofSetColor(ofColor::fireBrick, 128);
	ofDrawArrow(pos, pos + mParticleSystem->getGravity().normalized() * arrowLength, 0.02f);
	
	ofSetColor(ofColor::darkGrey, 64);
	ofDrawArrow(pos, pos + ofVec3f(0, -arrowLength, 0), 0.01f);
	// *** dg

	mMainCamera.end();

	

	// draw HUD
	mHud.draw();

	ofFill();
	float opacity = std::min(mTextDuration, 1.0f);
	if (opacity > 0.f)
	{
		ofSetColor(ofColor::aliceBlue, int(opacity * 255));
		mMainFont.drawString(mMainString, ofGetWindowWidth() * 0.3f, ofGetWindowHeight() * 0.2f);
	}
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){
	switch (key)
	{
	case OF_KEY_ESC: quit();
		break;
	case 'w':
		mMoveVec.z = .5f;
		break;
	case 's':
		mMoveVec.z = -.5f;
		break;
	case 'a':
		mMoveVec.x = .5f;
		break;
	case 'd':
		mMoveVec.x = -.5f;
		break;
	case 'v':
		//mValve = true;
		break;
	default:
		break;
	}
}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){
	switch (key)
	{
		case 'w': //fallthrough
		case 's':
			mMoveVec.z = 0.f;
			break;
		case 'a': //fallthrough
		case 'd':
			mMoveVec.x = 0.f;
			break;
		case 'h':
			std::cout << "Camera:" << mMainCamera.getPosition() << std::endl;
			break;
		case 'r':
			//mParticleSystem->setNumberOfParticles(0);
			mLevel.resetLevel();
			mHudGameHelp = HELP_START;
			break;
		case 'n':
			mGlobalRotation.normalize();
			mParticleSystem->setRotation(mGlobalRotation);
			mHudRotation = mGlobalRotation;
			break;
		case 't':
			//std::cout << mParticleSystem->debug_testIfParticlesOutside() << "\n";
			mParticleSystem->measureNextUpdate();
			break; 
		case 'u':
			mParticleSystem->update(0.16f);
			break;
		case 'e':
		{
			Level::GameState prev = mLevel.getGameState();

			mLevel.start();

			if (prev == Level::GameState::Ready && mLevel.isRunning())
			{
				mMainString = "START";
				mTextDuration = 1.0f;
				mHudGameHelp = HELP_RESET;
			}
			//ofVec3f tmpSize = mParticleSystem->getDimensions() * 0.5f;
			//mParticleSystem->addCube(tmpSize * ofVec3f(ofRandom(1.0f), 1, ofRandom(1.0f)), tmpSize, mXmlSettings.getValue("GENERAL:DROPSIZE", 1000));
			//mParticleSystem->addCube(ofVec3f(0, 0.5f, 0), tmpSize, mXmlSettings.getValue("GENERAL:DROPSIZE", 1000));
		}
			break;
		case 'v':
			//mValve = false;
			/*{
				ofVec3f tmpSize = mParticleSystem->getDimensions() * 0.5f;
				mParticleSystem->addCube(ofVec3f(0, tmpSize.y, 0), tmpSize * ofVec3f(0.5f, 1.f, 0.5f), 500, true);
			}*/
			break;
		case 'm':
		{
			ParticleSystem::ComputeMode currentMode = mParticleSystem->nextMode(mParticleSystem->getMode());
			mParticleSystem->setMode(currentMode);
			mHudMode = iHudGetModeString(currentMode);
		}
			break;
		case 'y':	mParticleSystem->toggleGenericSwitch();
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
			mMainCamera.setPosition(mMainCamera.getPosition() + ofVec3f(0, .01f * sens * mMouseSens * (y - mMouse.y), 0));
			//std::cout << "X ";
		}

		if (mRotationAxis & 0b010)
		{
			//ofQuaternion xRotation(sens * mMouseSens * (y - mMouse.y), ofVec3f(-1, 0, 0));
			float angle = sens * mMouseSens * (x - mMouse.x);
			ofQuaternion yRotation(angle, ofVec3f(0, 1, 0));
			mGlobalRotation *= yRotation;
			
			mSunDirection.rotate(-angle, ofVec3f(0, 1, 0));
			//mParticleSystem->setRotation(mGlobalRotation);

			mHudRotation = mGlobalRotation;
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
	switch (m)
	{
	case ParticleSystem::CPU:
			return "CPU";
		break;
	case ParticleSystem::COMPUTE_SHADER:
			return "Compute Shader";
		break;
	case ParticleSystem::CUDA:
		return "CUDA";
		break;
	case ParticleSystem::OPENCL:
			return "OpenCL";
		break;
	default:
			return "UNKNOWN";
		break;
	}
}
