#pragma once

#include "ofMain.h"
#include "ofxGui.h"
#include "ofxXmlSettings.h"

#include "Cube.h"
#include "vec2i.h"
//#include "ParticleSystem.h"
#include "CollisionSystem.h"

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

		ofxXmlSettings mXmlSettings;

		ofxPanel mHud;

		ofParameterGroup mHudDebugGroup;
		ofParameter<float> mHudFps;
		ofParameter<ofQuaternion> mHudRotation;

		ofParameterGroup mHudControlGroup;
		ofParameter<std::string> mHudMode;
		ofParameter<size_t> mHudWorkGroup;
		ofParameter<bool> mHudPause;
		ofParameter<bool> mHudStep;
		ofParameter<bool> mHudDraw;
		ofParameter<bool> mHudMovement;
		ofParameter<bool> mHudCollision;

		std::string iHudGetModeString(CollisionSystem::ComputeMode m);

		CollisionSystem mCollisionSystem;
		std::vector<int> mCollisions;
};
