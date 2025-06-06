#include "simulation.hpp"
#include "simulation.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>

// Device pointers
static Particle* d_particles = nullptr;
static float* d_forceMatrix = nullptr;
static float* d_influenceRadiusMatrix = nullptr;
static int particleCount = 0;
static int typeCount = 0;

// CUDA kernel
__global__ void launch_simulation_kernel(Particle* particles,
                                            float* forceMatrix,
                                            float* influenceRadiusMatrix,
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

        
        // printf("px: %f, py: %f, r: %f\n", p.x, p.y, influenceRadius);

        if (dist >= influenceRadius || dist <= 1e-3f) continue;

        float force = forceMatrix[p.type * typeCount + q.type] / r2;
        force *= 100;
        force = fminf(force, MAX_FORCE);

        // force = force*100.0f;
        fx += force * dx / dist;
        fy += force * dy / dist;
    }

    // p.vx += fx * dt;
    // p.vy += fy * dt;
    p.vx = (p.vx + fx * dt) * 0.9f;
    p.vy = (p.vy + fy * dt) * 0.9f;
    
    p.x += p.vx * dt;
    p.y += p.vy * dt;

    // Bounce off left/right
    if (p.x < 0.0f) {
        p.x = 0.0f;
        p.vx *= -1.0f;
    }
    if (p.x > screenWidth - 1) {
        p.x = screenWidth - 1;
        p.vx *= -1.0f;
    }

    // Bounce off top/bottom
    // if (p.y < 0.0f) {
    //     p.y = 0.0f;
    //     p.vy *= -1.0f;
    // }
    // if (p.y > screenHeight - 1) {
    //     p.y = screenHeight - 1;
    //     p.vy *= -1.0f;
    // }

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


    particles[i] = p;
}

// Host-side wrapper
void initializeSimulation(int numParticles,
                            int numTypes,
                            float* hostForceMatrix,
                            float* hostInfluenceRadiusMatrix) {
    particleCount = numParticles;
    typeCount = numTypes;
    cudaMalloc(&d_particles, particleCount * sizeof(Particle));
    cudaMalloc(&d_forceMatrix, typeCount * typeCount * sizeof(float));
    cudaMalloc(&d_influenceRadiusMatrix, typeCount * typeCount * sizeof(float));
    cudaMemcpy(d_forceMatrix, hostForceMatrix, typeCount * typeCount * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_influenceRadiusMatrix, hostInfluenceRadiusMatrix, typeCount * typeCount * sizeof(float), cudaMemcpyHostToDevice);
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
