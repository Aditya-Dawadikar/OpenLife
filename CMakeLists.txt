cmake_minimum_required(VERSION 3.10)
project(CUDAParticles LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

find_package(SDL2 REQUIRED)

include_directories(
    include
    ${SDL2_INCLUDE_DIRS}
)

file(GLOB SRC_FILES
    src/*.cpp
    src/*.cu
)

add_executable(openlife ${SRC_FILES})
target_link_libraries(openlife ${SDL2_LIBRARIES})
