#pragma once
#include "particle.hpp"

/**
 * Initializes the simulation on the GPU.
 * Allocates memory for particles and the force matrix, and uploads the matrix.
 *
 * @param numParticles Number of particles
 * @param numTypes Number of particle types (colors/species)
 * @param hostForceMatrix Flattened [numTypes * numTypes] force matrix (row-major)
 */
void initializeSimulation(int numParticles, int numTypes, float* hostForceMatrix, float* influenceRadiusMatrix, float* repulsionRadiusMatrix);

/**
 * Performs one simulation step on the GPU.
 *
 * @param dt Timestep to integrate over
 */
void runSimulationStep(float dt, int screenWidth, int screenHeight);

/**
 * Releases all GPU memory used by the simulation.
 */
void cleanupSimulation();

/**
 * Uploads particles from host (CPU) to device (GPU).
 *
 * @param hostParticles Pointer to host particle array
 * @param count Number of particles to upload
 */
void uploadParticles(const Particle* hostParticles, int count);

/**
 * Downloads particles from device (GPU) to host (CPU).
 *
 * @param hostParticles Pointer to host array to write into
 * @param count Number of particles to download
 */
void downloadParticles(Particle* hostParticles, int count);

// Update device-side force matrix
void updateForceMatrix(float* hostForceMatrix);

// Update device-side influence radius matrix
void updateInfluenceRadiusMatrix(float* hostRadiusMatrix);

// Update device-side repulsion radius matrix
void updateRepulsionRadiusMatrix(float* hostRadiusMatrix);
