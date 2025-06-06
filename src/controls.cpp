#include "controls.hpp"
#include <string>

bool simulationStarted = false;
bool simulationPaused = false;
bool simulationResetRequested = false;
int particleCountSetting = 8000;  // default count

// Exposed state
float forceMatrix[NUM_TYPES * NUM_TYPES] = {
    2.0f,  2.5f,  0.0f,
    2.5f, -2.0f, -1.2f,
   -0.4f, -0.6f, -0.1f
};

float influenceRadiusMatrix[NUM_TYPES * NUM_TYPES] = {
    60.0f, 80.0f, 100.0f,
    80.0f, 100.0f, 60.0f,
    100.0f, 60.0f, 70.0f
};

float particleDensity[NUM_TYPES] = {
    0.25f, 0.25f, 0.5f
};

void renderControlUI() {
    ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(300, 600), ImGuiCond_Always);

    // ImGui::Begin("Simulation Controls");
    ImGui::Begin("Simulation Controls", nullptr,
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoCollapse
    );


    if (!simulationStarted) {
        // if (ImGui::Button("Start Simulation")) {
        //     simulationStarted = true;
        // }
        ImGui::InputInt("Particles", &particleCountSetting);
        particleCountSetting = std::max(100, std::min(100000, particleCountSetting));

        if (ImGui::Button("Start Simulation")) {
            simulationStarted = true;
        }
    } else {
        // ImGui::Text("Running...");
        if (ImGui::Button(simulationPaused ? "▶ Play" : "⏸ Pause")) {
            simulationPaused = !simulationPaused;
        }

        if (ImGui::Button("🔁 Reset")) {
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

    // Force sliders
    // for (int i = 0; i < NUM_TYPES; ++i) {
    //     for (int j = i; j < NUM_TYPES; ++j) {
    //         // std::string label = "Force[" + std::to_string(i) + "][" + std::to_string(j) + "]";
    //         const char* typeNames[NUM_TYPES] = { "R", "G", "B" };
    //         std::string label = "F [" + std::string(typeNames[i]) + " <-> " + std::string(typeNames[j]) + "]";
    //         float& ref = forceMatrix[i * NUM_TYPES + j];
    //         ImGui::SliderFloat(label.c_str(), &ref, -5.0f, 5.0f);

    //         // Mirror the value to ensure symmetry
    //         if (i != j) {
    //             forceMatrix[j * NUM_TYPES + i] = ref;
    //         }
    //     }
    // }
    for (int i = 0; i < NUM_TYPES; ++i) {
        for (int j = 0; j < NUM_TYPES; ++j) {
            const char* typeNames[NUM_TYPES] = { "R", "G", "B" };
            std::string label = "F [" + std::string(typeNames[i]) + " → " + std::string(typeNames[j]) + "]";
            ImGui::SliderFloat(label.c_str(), &forceMatrix[i * NUM_TYPES + j], -5.0f, 5.0f);
        }
    }


    ImGui::Separator();

    // Radius sliders
    for (int i = 0; i < NUM_TYPES; ++i) {
        for (int j = i; j < NUM_TYPES; ++j) {
            // std::string label = "Radius[" + std::to_string(i) + "][" + std::to_string(j) + "]";
            const char* typeNames[NUM_TYPES] = { "R", "G", "B" };
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
