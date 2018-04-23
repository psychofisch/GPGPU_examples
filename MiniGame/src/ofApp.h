#pragma once

#include "ofMain.h"
#include "ofxGui.h"
#include "ofxXmlSettings.h"

#include "ParticleSystem.h"

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
		of3dPrimitive mTestBox;
		//ofEasyCam mMainCamera;
		ofCamera mMainCamera;
		ofLight mLight;
		//ofMesh mParticleMesh;
		ParticleSystem *mParticleSystem;
		int mRotationAxis;
		ofQuaternion mGlobalRotation;
		ofVec2f mMouse;
		float mMouseSens;
		ofVec3f mMoveVec;
		bool mValve;

		ofxXmlSettings mXmlSettings;

		ofxPanel mHud;

		ofParameterGroup mHudDebugGroup;
		ofParameter<float> mHudFps;
		ofParameter<ofQuaternion> mHudRotation;

		ofParameterGroup mHudControlGroup;
		ofParameter<std::string> mHudMode;
		ofParameter<size_t> mHudWorkGroup;
		ofParameter<std::string> mHudParticles;
		ofParameter<bool> mHudPause;
		ofParameter<bool> mHudStep;
		ofParameter<ofColor> mHudColor;

		ofParameterGroup mHudSimulationGroup;
		ofParameter<float> mHudSmoothingWidth;
		ofParameter<float> mHudRestDensity;
		ofParameter<float> mHudSpring;
		ofParameter<float> mHudSpringNear;

		std::string iHudGetModeString(ParticleSystem::ComputeMode m);
};
