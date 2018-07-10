#pragma once

#include "ofMain.h"
#include "ofxGui.h"
#include "ofxXmlSettings.h"

#include "Cube.h"
#include "vec2i.h"
//#include "ParticleSystem.h"
#include "CollisionSystem.h"

#define HANDLE_GL_ERROR() {GLenum err; while ((err = glGetError()) != GL_NO_ERROR) ofLogNotice() << __FILE__ << ":" << __LINE__ << ": GL error:	" << err;}

class ofApp : public ofBaseApp{

	public:
		void setup();
		void update();
		void draw();

		void keyPressed(int key);
		void keyReleased(int key);
		void mouseMoved(int x, int y );
		void mouseDragged(int x, int y, int button);
		void mousePressed(int x, int y, int button);
		void mouseReleased(int x, int y, int button);
		void mouseEntered(int x, int y);
		void mouseExited(int x, int y);
		void windowResized(int w, int h);
		void dragEvent(ofDragInfo dragInfo);
		void gotMessage(ofMessage msg);

		void resetCubes(int numberOfCubes);

		void quit();

	private:
		ofCamera mMainCamera;
		ofLight mLight;
		vec2i mMouse;
		vec2i mMoveVec;
		ofVec3f mCameraRotation;
		float mMouseSens;
		bool mLockMouse;
		std::vector<Cube> mCubes;
		CollisionSystem mCollisionSystem;
		std::vector<int> mCollisions;
		std::vector<ofVec4f> mCubePosAndSize;
		ofShader mBoxShader;
		ofBufferObject mPosAndSize;
		ofBufferObject mGPUCollisions;
		ofVboMesh mTemplateCube;

		ofxXmlSettings mXmlSettings;

		ofxPanel mHud;

		ofParameterGroup mHudDebugGroup;
		ofParameter<float> mHudFps;
		ofParameter<ofQuaternion> mHudRotation;

		ofParameterGroup mHudControlGroup;
		ofParameter<std::string> mHudMode;
		ofParameter<std::string> mHudCubes;
		ofParameter<size_t> mHudWorkGroup;
		ofParameter<bool> mHudPause;
		ofParameter<bool> mHudStep;
		ofParameter<bool> mHudDraw;
		ofParameter<bool> mHudMovement;
		ofParameter<bool> mHudCollision;

		std::string iHudGetModeString(CollisionSystem::ComputeMode m);
};
