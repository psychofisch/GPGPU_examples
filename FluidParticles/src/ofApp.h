#pragma once

#include "ofMain.h"
#include "ofxGui.h"
#include "ofxXmlSettings.h"

#include "ParticleSystem.h"
#include "Cube.h"

class ofApp : public ofBaseApp {

public:
	void setup();
	void update();
	void draw();

	void keyPressed(int key);
	void keyReleased(int key);
	void mouseMoved(int x, int y);
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
	std::shared_ptr<ParticleSystem> mParticleSystem;
	int mRotationAxis;
	ofQuaternion mGlobalRotation;
	ofVec2f mMouse;
	float mMouseSens;
	ofVec3f mMoveVec;
	bool mValve;
	ofVec3f mSunDirection;
	ofShader mWorldShader;
	ofBoxPrimitive mWorld;
	std::vector<Cube> mCollider;
	float mMeasureTime;

	ofxXmlSettings mXmlSettings;

	ofTrueTypeFont mMainFont;
	float mTextDuration;
	string mMainString;

	ofxPanel mHud;

	ofParameterGroup mHudDebugGroup;
	ofParameter<float> mHudFps;
	ofParameter<ofQuaternion> mHudRotation;

	ofParameterGroup mHudControlGroup;
	ofParameter<std::string> mHudMode;
	ofParameter<size_t> mHudWorkGroup;
	ofParameter<std::string> mHudParticles;
	ofParameter<float> mHudTime;
	ofParameter<float> mHudLastUpdate;
	ofParameter<bool> mHudSaveOnExit;

	ofParameterGroup mHudSimulationGroup;
	ofParameter<bool> mHudPause;
	ofParameter<bool> mHudStep;
	ofParameter<float> mHudInteractionRadius;
	ofParameter<float> mHudPressureMultiplier;
	ofParameter<float> mHudViscosity;
	ofParameter<float> mHudRestPressure;
};
