#pragma once
#include <SDL2/SDL.h>
#include "particle.hpp"

class Renderer {
public:
    Renderer(int width, int height);
    ~Renderer();

    void clear();
    void drawParticles(const Particle* particles, int count);
    void present();

private:
    SDL_Window* window;
    SDL_Renderer* renderer;
    int screenWidth, screenHeight;

    SDL_Color getColor(int type);
};
