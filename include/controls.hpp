#pragma once
#include <imgui.h>

// const int NUM_TYPES = 1;
// const int NUM_TYPES = 2;
// const int NUM_TYPES = 3;
// const int NUM_TYPES = 6;
const int NUM_TYPES = 10;

extern float forceMatrix[NUM_TYPES * NUM_TYPES];
extern float influenceRadiusMatrix[NUM_TYPES * NUM_TYPES];
extern float repulsionRadiusMatrix[NUM_TYPES * NUM_TYPES];
extern float particleDensity[NUM_TYPES];

extern bool simulationStarted;
extern bool simulationPaused;
extern bool simulationResetRequested;
extern int particleCountSetting;

void renderControlUI();
