#include "renderer.hpp"
#include <stdexcept>

Renderer::Renderer(int width, int height)
    : screenWidth(width), screenHeight(height) {

    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        throw std::runtime_error("SDL could not initialize");
    }

    // window = SDL_CreateWindow("OpenLife Simulation",
    //                           SDL_WINDOWPOS_CENTERED,
    //                           SDL_WINDOWPOS_CENTERED,
    //                           screenWidth,
    //                           screenHeight,
    //                           SDL_WINDOW_SHOWN);

    window = SDL_CreateWindow("OpenLife Simulation",
                          SDL_WINDOWPOS_CENTERED,
                          SDL_WINDOWPOS_CENTERED,
                          screenWidth,  // add space for UI
                          screenHeight,
                          SDL_WINDOW_SHOWN);


    if (!window) {
        throw std::runtime_error("SDL window could not be created");
    }

    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (!renderer) {
        throw std::runtime_error("SDL renderer could not be created");
    }
}

Renderer::~Renderer() {
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
}

void Renderer::clear() {
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255); // Black background
    SDL_RenderClear(renderer);
}

void Renderer::present() {
    SDL_RenderPresent(renderer);
}

SDL_Color Renderer::getColor(int type) {
    // Define color per type (can expand if more types added)
    switch (type) {
        case 0: return {255, 0, 0};       // Red
        case 1: return {0, 255, 0};       // Green
        case 2: return {0, 0, 255};       // Blue
        case 3: return {255, 255, 0};     // Yellow
        case 4: return {255, 0, 255};     // Magenta
        case 5: return {0, 255, 255};     // Cyan
        case 6: return {255, 165, 0};     // Orange
        case 7: return {255, 255, 255};   // White
        case 8: return {255, 105, 180};   // Pink
        case 9: return {139, 69, 19};     // Brown
        default: return {200, 200, 200};  // Light gray fallback
    }
}

void Renderer::drawParticles(const Particle* particles, int count) {
    for (int i = 0; i < count; ++i) {
        const Particle& p = particles[i];

        // SDL_RenderFillRect(renderer, &pixel);
        if (p.x >= 300 && p.x < screenWidth + 300 && p.y >= 0 && p.y < screenHeight) {
            SDL_Color color = getColor(p.type);
            SDL_SetRenderDrawColor(renderer, color.r, color.g, color.b, 255);

            SDL_Rect pixel = {
                static_cast<int>(p.x),
                static_cast<int>(p.y),
                2, 2
            };

            SDL_RenderFillRect(renderer, &pixel);
        }
    }
}

SDL_Renderer* Renderer::getSDLRenderer() const { return renderer; }
SDL_Window* Renderer::getWindow() const { return window; }