#pragma once

#ifndef HANDLE_GL_ERROR()
#define HANDLE_GL_ERROR() {GLenum err; while ((err = glGetError()) != GL_NO_ERROR) ofLogWarning() << __FILE__ << ":" << __LINE__ << ": GL error = " << err;}
#endif

#include "ofMain.h"
#include "ofxGui.h"
#include "ofxXmlSettings.h"

#include "Box.h"
#include "vec2i.h"
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

		void resetBoxes(int numberOfBoxes);

		void quit();

	private:
		ofCamera mMainCamera;
		ofLight mLight;
		vec2i mMouse;
		vec2i mMoveVec;
		ofVec3f mCameraRotation;
		float mMouseSens;
		bool mLockMouse;
		std::vector<Box> mBoxes;
		CollisionSystem mCollisionSystem;
		std::vector<int> mCollisions;
		std::vector<ofVec4f> mBoxPosAndSize;
		ofShader mBoxShader;
		ofBufferObject mPosAndSize;
		ofBufferObject mGPUCollisions;
		ofVboMesh mTemplateBox;
		//float mTargetCollisions;

		ofxXmlSettings mXmlSettings;

		ofxPanel mHud;

		ofParameterGroup mHudDebugGroup;
		ofParameter<float> mHudFps;
		ofParameter<std::string> mHudCollisionPercentage;
		ofParameter<bool> mHudMeasureNext;
		ofParameter<float> mHudMeasureTime;
		float mAutoMeasure;

		ofParameterGroup mHudControlGroup;
		ofParameter<std::string> mHudMode;
		ofParameter<std::string> mHudBoxes;
		ofParameter<bool> mHudDraw;
		ofParameter<bool> mHudMovement;
		ofParameter<bool> mHudCollision;
};
