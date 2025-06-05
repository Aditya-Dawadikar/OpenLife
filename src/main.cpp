#include <vector>
#include <cstdlib>
#include <ctime>

#include "renderer.hpp"
#include "simulation.hpp"
#include "particle.hpp"

const int WIDTH = 800;
const int HEIGHT = 600;
const int NUM_PARTICLES = 3000;
const int NUM_TYPES = 3;
const float DT = 0.1f;

int main(int argc, char* argv[]) {
    srand(static_cast<unsigned>(time(nullptr)));

    // Init Renderer
    Renderer renderer(WIDTH, HEIGHT);

    // Define Force Matrix (row-major: from_type * NUM_TYPES + to_type)
    float forceMatrix[NUM_TYPES * NUM_TYPES] = {
        2.0f,  1.5f,  0.0f,    // Red nucleus
        1.5f, -2.0f, -1.2f,    // Green membrane
        -0.3f, -0.6f, -0.1f    // Blue environment
    };

    float influenceRadiusMatrix[NUM_TYPES*NUM_TYPES] = {
        60.0f, 40.0f, 90.0f,   // Red: tight interaction
        40.0f, 80.0f, 40.0f,   // Green: thinner membrane
        90.0f, 40.0f, 60.0f    // Blue: not very influential
    };

    float particleDensity[NUM_TYPES] = {0.3f, 0.2f, 0.5f};

    // Create Particles
    std::vector<Particle> particles;
    for (int type = 0; type < NUM_TYPES; ++type) {
        int count = static_cast<int>(particleDensity[type]*NUM_PARTICLES);
        for (int i = 0; i < count; ++i) {
            Particle p;
            p.x = static_cast<float>(rand() % WIDTH);
            p.y = static_cast<float>(rand() % HEIGHT);
            p.vx = 0.0f;
            p.vy = 0.0f;
            p.type = type;
            particles.push_back(p);
        }
    }

    // Setup Simulation
    initializeSimulation(particles.size(),
                            NUM_TYPES,
                            forceMatrix,
                            influenceRadiusMatrix);
    uploadParticles(particles.data(), particles.size());

    // Main Loop
    bool running = true;
    SDL_Event event;

    bool paused = false;

    while (running) {
        // Handle events
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT)
                running = false;
            if (event.type == SDL_KEYDOWN) {
                if (event.key.keysym.sym == SDLK_SPACE) {
                    paused = !paused;
                }
            }
        }

        if (!paused) {
            // Step simulation
            runSimulationStep(DT, WIDTH, HEIGHT);
            downloadParticles(particles.data(), particles.size());

            // Render
            renderer.clear();
            renderer.drawParticles(particles.data(), particles.size());
            renderer.present();
        }

        SDL_Delay(16); // ~60 FPS
    }

    cleanupSimulation();
    return 0;
}
