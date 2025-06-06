#include <vector>
#include <cstdlib>
#include <ctime>

#include "renderer.hpp"
#include "simulation.hpp"
#include "particle.hpp"
#include "controls.hpp"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_sdlrenderer2.h"

const int WIDTH = 1200;
const int HEIGHT = 600;
const float DT = 0.1f;

int main(int argc, char* argv[]) {
    srand(static_cast<unsigned>(time(nullptr)));

    // Init Renderer
    Renderer renderer(WIDTH, HEIGHT);

    // Setup ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();

    ImGui_ImplSDL2_InitForSDLRenderer(renderer.getWindow(), renderer.getSDLRenderer());
    ImGui_ImplSDLRenderer2_Init(renderer.getSDLRenderer());

    std::vector<Particle> particles;
    bool simulationInitialized = false;

    // Main Loop
    bool running = true;
    SDL_Event event;

    while (running) {
        // Handle Reset
        if (simulationResetRequested) {
            cleanupSimulation();
            particles.clear();
            simulationInitialized = false;
            simulationResetRequested = false;
        }

        // Handle Events
        while (SDL_PollEvent(&event)) {
            ImGui_ImplSDL2_ProcessEvent(&event);
            if (event.type == SDL_QUIT)
                running = false;
            if (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_SPACE)
                simulationPaused = !simulationPaused;
        }

        // Start ImGui Frame
        ImGui_ImplSDL2_NewFrame();
        ImGui_ImplSDLRenderer2_NewFrame();
        ImGui::NewFrame();

        // UI panel
        renderControlUI();

        // Initialize Simulation (only once after "Start")
        if (simulationStarted && !simulationInitialized) {
            particles.clear();
            for (int type = 0; type < NUM_TYPES; ++type) {
                int count = static_cast<int>(particleDensity[type] * particleCountSetting);
                for (int i = 0; i < count; ++i) {
                    Particle p;
                    p.x = 325 + static_cast<float>(rand() % 600);  // Right-shift to leave space for UI
                    p.y = static_cast<float>(rand() % 600);
                    p.vx = 0.0f;
                    p.vy = 0.0f;
                    p.type = type;
                    particles.push_back(p);
                }
            }

            initializeSimulation(particles.size(), NUM_TYPES, forceMatrix, influenceRadiusMatrix);
            uploadParticles(particles.data(), particles.size());
            simulationInitialized = true;
        }

        // Step Simulation if running
        if (simulationStarted && simulationInitialized && !simulationPaused) {
            updateForceMatrix(forceMatrix);
            updateInfluenceRadiusMatrix(influenceRadiusMatrix);

            runSimulationStep(DT, WIDTH, HEIGHT);
            downloadParticles(particles.data(), particles.size());
        }

        // Render
        renderer.clear();
        if (simulationInitialized)
            renderer.drawParticles(particles.data(), particles.size());

        ImGui::Render();
        ImGui_ImplSDLRenderer2_RenderDrawData(ImGui::GetDrawData(), renderer.getSDLRenderer());
        renderer.present();

        SDL_Delay(16); // ~60 FPS
    }

    // Cleanup
    cleanupSimulation();
    ImGui_ImplSDLRenderer2_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();
    return 0;
}
