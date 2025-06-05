#pragma once
#include "particle.hpp"

// Launch the CUDA kernel to update particle positions
__global__ void launch_simulation_kernel(Particle* particles,
                                            float* forceMatrix,
                                            float* influenceRadiusMatrix,
                                            int count,
                                            int typeCount,
                                            float dt,
                                            int screenWidth,
                                            int screenHeight);
