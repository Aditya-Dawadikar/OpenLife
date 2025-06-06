#include "controls.hpp"
#include <string>

bool simulationStarted = false;
bool simulationPaused = false;
bool simulationResetRequested = false;
int particleCountSetting = 8000;  // default count

// const char* typeNames[NUM_TYPES] = { "R", "G", "B" };
const char* typeNames[NUM_TYPES] = { "R", "G", "B", "Y", "M", "C" };

// float forceMatrix[NUM_TYPES * NUM_TYPES] = {
//     -1.5f, 1.0f, 2.0f,
//     1.0f, -0.5f, 1.5f,
//     2.0f, 1.5f, -1.0f
// };
float forceMatrix[6 * 6] = {
    //  O      C      H      P      N      S
    -1.5,  +1.0,  +2.0,  +1.5,  +1.0,  -0.5,  // O
    +1.0,  -0.5,  +1.5,  +1.0,  +1.5,  +1.0,  // C
    +2.0,  +1.5,  -1.0,  +1.0,  +2.0,  +1.0,  // H
    +1.5,  +1.0,  +1.0,  -0.5,  +1.0,  -0.5,  // P
    +1.0,  +1.5,  +2.0,  +1.0,  -0.5,  +0.5,  // N
    -0.5,  +1.0,  +1.0,  -0.5,  +0.5,  -1.0   // S
};

// float influenceRadiusMatrix[NUM_TYPES * NUM_TYPES] = {
//     60.0f, 90.0f, 100.0f,
//     90.0f, 80.0f, 90.0f,
//     100.0f, 90.0f, 60.0f
// };
float influenceRadiusMatrix[6 * 6] = {
    // O     C     H     P     N     S
     60,   90,  100,   80,   80,   60,  // O
     90,   80,   90,   70,   85,   80,  // C
    100,   90,   60,   70,   90,   60,  // H
     80,   70,   70,   60,   60,   50,  // P
     80,   85,   90,   60,   70,   60,  // N
     60,   80,   60,   50,   60,   60   // S
};

// float particleDensity[NUM_TYPES] = {
//     0.15f, 0.25f, 0.60f 
// };
// Oxygen, Carbon, Hydrogen, Phosphorus, Nitrogen, Sulfur
float particleDensity[6] = {
    0.12f,  // O
    0.20f,  // C
    0.40f,  // H
    0.05f,  // P
    0.15f,  // N
    0.08f   // S
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

    // Density sliders
    for (int i = 0; i < NUM_TYPES; ++i) {
        std::string label = "D [" + std::to_string(i) + "]";
        ImGui::SliderFloat(label.c_str(), &particleDensity[i], 0.0f, 1.0f);
    }

    ImGui::Separator();

    for (int i = 0; i < NUM_TYPES; ++i) {
        for (int j = 0; j < NUM_TYPES; ++j) {
            std::string label = "F [" + std::string(typeNames[i]) + " → " + std::string(typeNames[j]) + "]";
            ImGui::SliderFloat(label.c_str(), &forceMatrix[i * NUM_TYPES + j], -5.0f, 5.0f);
        }
    }


    ImGui::Separator();

    // Radius sliders
    for (int i = 0; i < NUM_TYPES; ++i) {
        for (int j = i; j < NUM_TYPES; ++j) {
            std::string label = "R [" + std::string(typeNames[i]) + " <-> " + std::string(typeNames[j]) + "]";
            float& ref = influenceRadiusMatrix[i * NUM_TYPES + j];
            ImGui::SliderFloat(label.c_str(), &ref, 10.0f, 200.0f);

            if (i != j) {
                influenceRadiusMatrix[j * NUM_TYPES + i] = ref;
            }
        }
    }

    ImGui::End();
}
