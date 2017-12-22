#pragma once

#include "ofMain.h"
#include "ofxGui.h"

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
		float mLastFrame;
		ofBoxPrimitive mTestBox;
		//ofEasyCam mMainCamera;
		ofCamera mMainCamera;
		ofLight mLight;
		//ofMesh mParticleMesh;
		ofVbo mParticlesVBO;
		ParticleSystem *mParticleSystem;
		int mRotationAxis;
		ofQuaternion mGlobalRotation;
		ofVec2f mMouse;
		float mMouseSens;
		bool mValve;

		ofxPanel mHud;
		ofxLabel mHudFps;
		ofxLabel mHudRotation;
		ofxColorSlider mHudColor;
};
