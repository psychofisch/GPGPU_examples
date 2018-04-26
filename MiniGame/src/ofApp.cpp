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

	ofGetWindowPtr()->setWindowTitle(mXmlSettings.getValue("GENERAL:TITLE", "Minigame"));

	ofSetVerticalSync(false);

	ofBackground(69, 69, 69);

	//mTestBox.setResolution(1);
	//mTestBox.setScale(1.f);
	mTestBox.setPosition(ofVec3f(0.f));

	mMainCamera.setPosition(0, 0, 2);
	mMainCamera.lookAt(ofVec3f(0.f));
	mMainCamera.setNearClip(0.01f);
	mMainCamera.setFarClip(50.f);

	mLight.setPointLight();
	mLight.setPosition(ofVec3f(0.f));

	ofBoxPrimitive testRect;

	int maxParticles = mXmlSettings.getValue("GENERAL:MAXPARTICLES", 5000);
	if (maxParticles <= 0)
	{
		std::cout << "WARNING: GENERAL:MAXPARTICLES was \"<= 0\" again.\n";
		maxParticles = 5000;
	}
	mParticleSystem = new ParticleSystem(maxParticles);
	mParticleSystem->setupAll(mXmlSettings);
	mParticleSystem->setMode(ParticleSystem::ComputeMode::CPU);
	mParticleSystem->setDimensions(ofVec3f(1.f));
	Particle::CUDAta tmpCUDA = mParticleSystem->getCudata();
	mParticleSystem->createParticleShader(mXmlSettings.getValue("GENERAL:VERT", "shader.vert"), mXmlSettings.getValue("GENERAL:FRAG", "shader.frag"));
	//mParticleSystem->addDamBreak(200);
	//mParticleSystem->addCube(ofVec3f(0), mParticleSystem->getDimensions(), 200);
	//mParticleSystem->addRandom();

	mValve = false;

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
	mHudControlGroup.add(mHudColor.set("Particle Color", ofColor(100, 100, 140)));

	mHudSimulationGroup.setName("Simulation Settings");
	mHudControlGroup.add(mHudPause.set("Pause", false));
	mHudControlGroup.add(mHudStep.set("Step", false));
	mHudSimulationGroup.add(mHudSmoothingWidth.set("Interaction Radius", 0.1f, 0.00000001f, 1.f));
	mHudSimulationGroup.add(mHudRestDensity.set("Rest Density", 3.5f, 0.0f, 50.f));
	mHudSimulationGroup.add(mHudSpring.set("Spring", 1.0f, 0.0f, 10.f));
	mHudSimulationGroup.add(mHudSpringNear.set("Spring Near", 1.0f, 0.0f, 10.f));

	mHud.setup();
	mHud.add(mHudDebugGroup);
	mHud.add(mHudControlGroup);
	mHud.add(mHudSimulationGroup);
	mHud.loadFromFile("hud.xml");

	mHudMode = iHudGetModeString(mParticleSystem->getMode());

	std::cout << "OpenGL " << ofGetGLRenderer()->getGLVersionMajor() << "." << ofGetGLRenderer()->getGLVersionMinor() << std::endl;
}

//--------------------------------------------------------------
void ofApp::update(){
	float deltaTime =  std::min(0.1, ofGetLastFrameTime());
	//std::cout << deltaTime << std::endl;

	mHudFps = std::round(ofGetFrameRate());
	mHudParticles = ofToString(mParticleSystem->getNumberOfParticles()) + "/" + ofToString(mParticleSystem->getCapacity());
	//mHudFps = ofToString(ofGetFrameRate(),0) + "\t" + ofToString(mParticleSystem->getNumberOfParticles()) + "/" + ofToString(mParticleSystem->getCapacity());

	SimulationData simD;
	simD.interactionRadius = mHudSmoothingWidth;
	simD.rho0 = mHudRestDensity;
	simD.spring = mHudSpring;
	simD.springNear = mHudSpringNear;
	mParticleSystem->setSimulationData(simD);
	//mParticleSystem->setSmoothingWidth(mHudSmoothingWidth);
	//mParticleSystem->setRestDensity(mHudRestDensity);
	//mParticleSystem->getCudata().maxWorkGroupSize = mHudWorkGroup;

	float spinX = sin(ofGetElapsedTimef()*.35f);
	float spinY = cos(ofGetElapsedTimef()*.075f);

	if (mValve)
	{
		ofVec3f tmpSize = mParticleSystem->getDimensions();
		mParticleSystem->addCube(tmpSize * ofVec3f(0.5f, 1.0f, 0.5f), tmpSize * 0.1f, 2, true);
	}

	ofVec3f moveVec = mMoveVec * deltaTime;
	mParticleSystem->addPosition(moveVec);
	mTestBox.move(moveVec);

	if (!mHudPause || mHudStep)
	{
		float dt = deltaTime;
		if (mHudStep)
			dt = 0.008f;
		mParticleSystem->update(dt);
		mHudStep = false;
	}
	//mParticleSystem->update(0.016f);
	//mParticleMesh.haveVertsChanged();
	//mTestBox.rotate(spinY, 0, 1, 0);
}

//--------------------------------------------------------------
void ofApp::draw(){
	ofEnableDepthTest();

	mMainCamera.lookAt(ofVec3f(0));
	//ofEnableLighting();
	//mLight.enable();
	mMainCamera.begin();

	//mTestBox.draw(ofPolyRenderMode::OF_MESH_WIREFRAME);
	
	//ofPushMatrix();
	ofVec3f axis;
	float angle;
	mGlobalRotation.getRotate(angle, axis);
	ofRotate(angle, axis.x, axis.y, axis.z);
	ofTranslate(-mParticleSystem->getDimensions() * 0.5f);
	//ofPushStyle();
	ofSetColor(mHudColor);
	//glPointSize(2.f);
	mTestBox.drawAxes(1.f);
	//mParticleMesh.drawVertices();
	//mParticlesVBO.draw(GL_POINTS, 0, mParticleSystem->getNumberOfParticles());
	mParticleSystem->draw();
	//ofPopStyle();
	//ofPopMatrix();

	mMainCamera.end();
	//mLight.disable();
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
	case 'w':
		mMoveVec.z = -1.f;
		break;
	case 's':
		mMoveVec.z = 1.f;
		break;
	case 'a':
		mMoveVec.x = -1.f;
		break;
	case 'd':
		mMoveVec.x = 1.f;
		break;
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
			std::cout << "Box: " << mTestBox.getPosition() << std::endl;
			break;
		case 'r':
			mParticleSystem->setNumberOfParticles(0);
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
			ofVec3f tmpSize = mParticleSystem->getDimensions() * 0.5f;
			mParticleSystem->addCube(tmpSize * ofVec3f(ofRandom(1.0f), 1, ofRandom(1.0f)), tmpSize, mXmlSettings.getValue("GENERAL:DROPSIZE", 1000));
		}
			break;
		case 'v':
			mValve = false;
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
			ofQuaternion yRotation(sens * mMouseSens * (x - mMouse.x), ofVec3f(0, 1, 0));
			mGlobalRotation *= yRotation;
			
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
