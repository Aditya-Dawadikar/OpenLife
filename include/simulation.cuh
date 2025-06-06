#pragma once
#include "particle.hpp"

// Launch the CUDA kernel to update particle positions

__global__ void launch_simulation_square_kernel(Particle* particles,
                                            float* forceMatrix,
                                            float* influenceRadiusMatrix,
                                            float* repulsionRadiusMatrix,
                                            int count,
                                            int typeCount,
                                            float dt,
                                            int screenWidth,
                                            int screenHeight);
__global__ void launch_simulation_kernel(Particle* particles,
                                            float* forceMatrix,
                                            float* influenceRadiusMatrix,
                                            float* repulsionRadiusMatrix,
                                            int count,
                                            int typeCount,
                                            float dt,
                                            int screenWidth,
                                            int screenHeight);
