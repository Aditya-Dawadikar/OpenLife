#include "particle.hpp"
#include <cstdlib>  // for rand()
#include <ctime>
#include <cmath>

Particle randomParticle(int screenWidth, int screenHeight, int numTypes, float* particleDensity) {
    Particle p;
    p.x = static_cast<float>(rand() % screenWidth);
    p.y = static_cast<float>(rand() % screenHeight);
    p.vx = 0.0f;
    p.vy = 0.0f;
    p.type = rand() % numTypes;

    return p;
}
