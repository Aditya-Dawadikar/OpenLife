#include "simulation.hpp"
#include "simulation.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>

// Device pointers
static Particle* d_particles = nullptr;
static float* d_forceMatrix = nullptr;
static float* d_influenceRadiusMatrix = nullptr;
static float* d_repulsionRadiusMatrix = nullptr;
static int particleCount = 0;
static int typeCount = 0;

// CUDA kernel - Square World
__global__ void launch_simulation_square_kernel(Particle* particles,
                                            float* forceMatrix,
                                            float* influenceRadiusMatrix,
                                            float* repulsionRadiusMatrix,
                                            int count,
                                            int typeCount,
                                            float dt,
                                            int screenWidth,
                                            int screenHeight) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    Particle p = particles[i];
    float fx = 0.0f, fy = 0.0f;

    float MAX_FORCE = 100.0f;

    for (int j = 0; j < count; ++j) {
        if (i == j) continue;
        Particle q = particles[j];

        float dx = q.x - p.x;
        float dy = q.y - p.y;
        float r2 = dx * dx + dy * dy + 1e-5f;
        float dist = sqrtf(r2);

        float influenceRadius = influenceRadiusMatrix[p.type*typeCount + q.type];
        float repulsionRadius = repulsionRadiusMatrix[p.type*typeCount + q.type];

        if (dist <= influenceRadius && dist > repulsionRadius){
            float force = forceMatrix[p.type * typeCount + q.type] / r2;
            force = fminf(force, MAX_FORCE);

            force = force*100.0f;
            fx += force * dx / dist;
            fy += force * dy / dist;
        }else if(dist <= repulsionRadius){
            float force = abs(forceMatrix[p.type * typeCount + q.type] / r2);
            force = fminf(force, MAX_FORCE);

            force = force*10.0f;
            fx -= force * dx / dist;
            fy -= force * dy / dist;
        }

    }

    p.vx = (p.vx + fx * dt) * 0.9f;
    p.vy = (p.vy + fy * dt) * 0.9f;
    
    p.x += p.vx * dt;
    p.y += p.vy * dt;

    // Define canvas bounds (hardcoded or passed in — here we hardcode)
    float canvasLeft = 325.0f;
    float canvasRight = canvasLeft + 600;  // screenWidth = 800

    if (p.x < canvasLeft) {
        p.x = canvasLeft;
        p.vx *= -1.0f;
    }
    if (p.x > canvasRight - 1.0f) {
        p.x = canvasRight - 1.0f;
        p.vx *= -1.0f;
    }
    if (p.y < 0.0f) {
        p.y = 0.0f;
        p.vy *= -1.0f;
    }
    if (p.y > screenHeight - 1.0f) {
        p.y = screenHeight - 1.0f;
        p.vy *= -1.0f;
    }


    particles[i] = p;
}

// CUDA Kernel - Spherical World
__global__ void launch_simulation_kernel(Particle* particles,
                                         float* forceMatrix,
                                         float* influenceRadiusMatrix,
                                         float* repulsionRadiusMatrix,
                                         int count,
                                         int typeCount,
                                         float dt,
                                         int screenWidth,
                                         int screenHeight) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    Particle p = particles[i];
    float fx = 0.0f, fy = 0.0f;
    float MAX_FORCE = 100.0f;

    float canvasLeft = 325.0f;
    float canvasWidth = 600.0f;
    float canvasRight = canvasLeft + canvasWidth;
    float canvasHeight = (float)screenHeight;

    for (int j = 0; j < count; ++j) {
        if (i == j) continue;
        Particle q = particles[j];

        float dx = q.x - p.x;
        float dy = q.y - p.y;

        // Toroidal wrap for X direction
        if (dx > canvasWidth / 2) dx -= canvasWidth;
        if (dx < -canvasWidth / 2) dx += canvasWidth;

        // Toroidal wrap for Y direction
        if (dy > canvasHeight / 2) dy -= canvasHeight;
        if (dy < -canvasHeight / 2) dy += canvasHeight;

        float r2 = dx * dx + dy * dy + 1e-5f;
        float dist = sqrtf(r2);

        float influenceRadius = influenceRadiusMatrix[p.type * typeCount + q.type];
        float repulsionRadius = repulsionRadiusMatrix[p.type * typeCount + q.type];

        if (dist <= influenceRadius && dist > repulsionRadius) {
            float force = forceMatrix[p.type * typeCount + q.type] / r2;
            force = fminf(force, MAX_FORCE);
            force *= 100.0f;

            fx += force * dx / dist;
            fy += force * dy / dist;
        } else if (dist <= repulsionRadius) {
            float force = fabsf(forceMatrix[p.type * typeCount + q.type] / r2);
            force = fminf(force, MAX_FORCE);
            force *= 10.0f;

            fx -= force * dx / dist;
            fy -= force * dy / dist;
        }
    }

    // Integrate and apply friction
    p.vx = (p.vx + fx * dt) * 0.9f;
    p.vy = (p.vy + fy * dt) * 0.9f;
    p.x += p.vx * dt;
    p.y += p.vy * dt;

    // Toroidal wrapping for position
    if (p.x < canvasLeft) p.x += canvasWidth;
    if (p.x >= canvasRight) p.x -= canvasWidth;
    if (p.y < 0.0f) p.y += canvasHeight;
    if (p.y >= canvasHeight) p.y -= canvasHeight;

    particles[i] = p;
}



// Host-side wrapper
void initializeSimulation(int numParticles,
                            int numTypes,
                            float* hostForceMatrix,
                            float* hostInfluenceRadiusMatrix,
                            float* hostRepulsionRadiusMatrix) {
    particleCount = numParticles;
    typeCount = numTypes;
    cudaMalloc(&d_particles, particleCount * sizeof(Particle));
    cudaMalloc(&d_forceMatrix, typeCount * typeCount * sizeof(float));
    cudaMalloc(&d_influenceRadiusMatrix, typeCount * typeCount * sizeof(float));
    cudaMalloc(&d_repulsionRadiusMatrix, typeCount * typeCount * sizeof(float));
    cudaMemcpy(d_forceMatrix, hostForceMatrix, typeCount * typeCount * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_influenceRadiusMatrix, hostInfluenceRadiusMatrix, typeCount * typeCount * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_repulsionRadiusMatrix, hostRepulsionRadiusMatrix, typeCount * typeCount * sizeof(float), cudaMemcpyHostToDevice);
}

void uploadParticles(const Particle* hostParticles, int count) {
    cudaMemcpy(d_particles, hostParticles, count * sizeof(Particle), cudaMemcpyHostToDevice);
}

void downloadParticles(Particle* hostParticles, int count) {
    cudaMemcpy(hostParticles, d_particles, count * sizeof(Particle), cudaMemcpyDeviceToHost);
}

void runSimulationStep(float dt, int screenWidth, int screenHeight) {
    int threadsPerBlock = 256;
    int blocks = (particleCount + threadsPerBlock - 1) / threadsPerBlock;
    launch_simulation_kernel<<<blocks, threadsPerBlock>>>(d_particles,
                                                            d_forceMatrix,
                                                            d_influenceRadiusMatrix,
                                                            d_repulsionRadiusMatrix,
                                                            particleCount,
                                                            typeCount,
                                                            dt,
                                                            screenWidth,
                                                            screenHeight);
    cudaDeviceSynchronize();
}

void cleanupSimulation() {
    cudaFree(d_particles);
    cudaFree(d_forceMatrix);
}


void updateForceMatrix(float* hostForceMatrix) {
    cudaMemcpy(d_forceMatrix, hostForceMatrix, typeCount * typeCount * sizeof(float), cudaMemcpyHostToDevice);
}

void updateInfluenceRadiusMatrix(float* hostRadiusMatrix) {
    cudaMemcpy(d_influenceRadiusMatrix, hostRadiusMatrix, typeCount * typeCount * sizeof(float), cudaMemcpyHostToDevice);
}

void updateRepulsionRadiusMatrix(float* hostRadiusMatrix) {
    cudaMemcpy(d_repulsionRadiusMatrix, hostRadiusMatrix, typeCount * typeCount * sizeof(float), cudaMemcpyHostToDevice);
}
