#pragma once

struct Particle {
    float x, y;     // Position
    float vx, vy;   // Velocity
    int type;       // Species/type ID
    // float radius;
};

// Optional helpers (declarations)
Particle randomParticle(int screenWidth, int screenHeight, int numTypes, float* particleDensity);
