#include "controls.hpp"
#include <string>

bool simulationStarted = false;
bool simulationPaused = false;
bool simulationResetRequested = false;
// int particleCountSetting = 8000;  // default count
int particleCountSetting = 2000;

// const char* typeNames[NUM_TYPES] = { "R" };
// const char* typeNames[NUM_TYPES] = { "R", "G", "B" };
// const char* typeNames[NUM_TYPES] = { "R", "G", "B", "Y", "M", "C" };
const char* typeNames[NUM_TYPES] = { "R", "G", "B", "Y", "M", "C", "O", "W", "P", "Br" };


// float forceMatrix[NUM_TYPES * NUM_TYPES] = {
//     0.0f  // R -> [R, R]
// };

// float forceMatrix[NUM_TYPES * NUM_TYPES] = {
//     0.0f,  0.3f,  0.9f,  // R -> [R, G, B]
//      -0.3f, 0.0f, 0.3f,  // G -> [R, G, B]
//      0.9f, -0.3f, 0.0f   // B -> [R, G, B]
// };

// float forceMatrix[NUM_TYPES * NUM_TYPES] = {
//     //  O      C      H      P      N      S
//     -1.5,  +1.0,  +2.0,  +1.5,  +1.0,  -0.5,  // O
//     +1.0,  -0.5,  +1.5,  +1.0,  +1.5,  +1.0,  // C
//     +2.0,  +1.5,  -1.0,  +1.0,  +2.0,  +1.0,  // H
//     +1.5,  +1.0,  +1.0,  -0.5,  +1.0,  -0.5,  // P
//     +1.0,  +1.5,  +2.0,  +1.0,  -0.5,  +0.5,  // N
//     -0.5,  +1.0,  +1.0,  -0.5,  +0.5,  -1.0   // S
// };

float forceMatrix[100] = {
    // R    G    B    Y    M    C    O    W    P    Br
     0.8, 0.2, 0.0, -0.4, 0.6, -0.2, 0.5, 1.0, 0.7, 0.1,  // R
     0.2, 0.7, 0.1, 0.4, -0.5, 0.3, 0.0, 0.75, 0.2, 0.0,  // G
     0.0, 0.1, 0.6, -0.3, 0.5, 0.2, -0.4, 0.5, 0.1, -0.2, // B
    -0.4, 0.4, -0.3, 0.9, 0.2, 0.0, 0.3, 0.25, -0.5, 0.3, // Y
     0.6,-0.5, 0.5, 0.2, 0.7, 0.4, -0.1, 0.0, 0.0, -0.4,  // M
    -0.2, 0.3, 0.2, 0.0, 0.4, 0.8, 0.1, -0.25, 0.5, 0.1,   // C
     0.5, 0.0,-0.4, 0.3,-0.1, 0.1, 0.9, -0.50, -0.2, 0.0,  // O
    -0.3,-0.6, -0.6,-0.1, -0.2,-0.3, -0.2, -1.0, -0.4, -0.1,   // W
     0.7, 0.2, 0.1,-0.5, 0.0, 0.5,-0.2, -0.75, 0.6, -0.3,  // P
     0.1, 0.0,-0.2, 0.3,-0.4, 0.1, 0.0, -1.0, -0.3, 0.8   // Br
};


// float influenceRadiusMatrix[NUM_TYPES * NUM_TYPES] = {100.0f};

// float influenceRadiusMatrix[NUM_TYPES * NUM_TYPES] = {
//     50.0f,  100.0f, 150.0f,
//     100.0f,  50.0f,  90.0f,
//     150.0f,  90.0f,  50.0f
// };

float influenceRadiusMatrix[100] = {
    80, 90, 100, 70, 120, 85, 60, 100, 110, 90,
    90, 80, 85, 95, 100, 75, 60, 90, 95, 80,
    100, 85, 80, 60, 110, 95, 70, 85, 80, 100,
    70, 95, 60, 80, 100, 90, 85, 75, 70, 85,
    120, 100,110,100, 90, 60, 80, 95, 105, 90,
    85, 75, 95, 90, 60, 70, 100, 80, 90, 85,
    60, 60, 70, 85, 80, 100, 90, 95, 70, 60,
    100, 90, 85, 75, 95, 80, 95, 85, 90, 100,
    110, 95, 80, 70,105, 90, 70, 90, 85, 95,
    90, 80,100, 85, 90, 85, 60,100, 95, 80
};


// float repulsionRadiusMatrix[NUM_TYPES * NUM_TYPES] = {5.0f};


// float repulsionRadiusMatrix[NUM_TYPES * NUM_TYPES] = {
//     20.0f,  20.0f, 20.0f,
//     20.0f,  20.0f, 20.0f,
//     20.0f,  20.0f, 20.0f
// };

float repulsionRadiusMatrix[100] = {
    20,20,20,20,20,20,20,20,20,20,
    20,20,20,20,20,20,20,20,20,20,
    20,20,20,20,20,20,20,20,20,20,
    20,20,20,20,20,20,20,20,20,20,
    20,20,20,20,20,20,20,20,20,20,
    20,20,20,20,20,20,20,20,20,20,
    20,20,20,20,20,20,20,20,20,20,
    20,20,20,20,20,20,20,20,20,20,
    20,20,20,20,20,20,20,20,20,20,
    20,20,20,20,20,20,20,20,20,20,
};


// float particleDensity[NUM_TYPES] = {
//     0.33f, 0.33f, 0.33f 
// };
// float particleDensity[NUM_TYPES] = {
//     1.0f 
// };
// // Oxygen, Carbon, Hydrogen, Phosphorus, Nitrogen, Sulfur
// float particleDensity[6] = {
//     0.12f,  // O
//     0.20f,  // C
//     0.40f,  // H
//     0.05f,  // P
//     0.15f,  // N
//     0.08f   // S
// };

float particleDensity[10] = {
    0.1f,  // R
    0.1f,  // G
    0.1f,  // B
    0.1f,  // Y
    0.1f,  // M
    0.1f,  // C
    0.1f,  // O
    0.1f,  // W
    0.1f,  // P
    0.1f   // Br
};


void renderControlUI() {
    ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(300, 600), ImGuiCond_Always);

    ImGui::Begin("Simulation Controls", nullptr,
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoCollapse
    );


    if (!simulationStarted) {
        ImGui::InputInt("Particles", &particleCountSetting);
        particleCountSetting = std::max(100, std::min(100000, particleCountSetting));

        if (ImGui::Button("Start Simulation")) {
            simulationStarted = true;
        }
    } else {
        if (ImGui::Button(simulationPaused ? "Play" : "Pause")) {
            simulationPaused = !simulationPaused;
        }

        if (ImGui::Button("Reset")) {
            simulationResetRequested = true;
            simulationStarted = false;
            simulationPaused = false;
        }

        ImGui::Text("Running...");
    }

    ImGui::Separator();

    ImGui::Text("Particle Density:");

    // Density sliders
    for (int i = 0; i < NUM_TYPES; ++i) {
        std::string label = "D [" + std::to_string(i) + "]";
        ImGui::SliderFloat(label.c_str(), &particleDensity[i], 0.0f, 1.0f);
    }

    ImGui::Separator();
    
    ImGui::Text("Force Strength:");

    for (int i = 0; i < NUM_TYPES; ++i) {
        for (int j = 0; j < NUM_TYPES; ++j) {
            std::string label = "F [" + std::string(typeNames[i]) + " -> " + std::string(typeNames[j]) + "]";
            ImGui::SliderFloat(label.c_str(), &forceMatrix[i * NUM_TYPES + j], -1.0f, 1.0f);
        }
    }


    ImGui::Separator();
    
    ImGui::Text("Attraction Radius:");

    // Radius sliders
    for (int i = 0; i < NUM_TYPES; ++i) {
        for (int j = i; j < NUM_TYPES; ++j) {
            std::string label = "A [" + std::string(typeNames[i]) + " >-< " + std::string(typeNames[j]) + "]";
            float& ref = influenceRadiusMatrix[i * NUM_TYPES + j];
            ImGui::SliderFloat(label.c_str(), &ref, 10.0f, 200.0f);

            if (i != j) {
                influenceRadiusMatrix[j * NUM_TYPES + i] = ref;
            }
        }
    }

    ImGui::Separator();
    
    ImGui::Text("Repulsion Radius:");
    for (int i = 0; i < NUM_TYPES; ++i) {
        for (int j = i; j < NUM_TYPES; ++j) {
            std::string label = "R [" + std::string(typeNames[i]) + " <-> " + std::string(typeNames[j]) + "]";
            float& ref = repulsionRadiusMatrix[i * NUM_TYPES + j];
            ImGui::SliderFloat(label.c_str(), &ref, 0.0f, 100.0f);

            if (i != j) {
                repulsionRadiusMatrix[j * NUM_TYPES + i] = ref;
            }
        }
    }

    ImGui::End();
}
