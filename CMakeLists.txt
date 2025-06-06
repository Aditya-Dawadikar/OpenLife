cmake_minimum_required(VERSION 3.10)
project(CUDAParticles LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(SDL2 REQUIRED)

# Add ImGui sources
set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/imgui)
set(IMGUI_BACKENDS ${IMGUI_DIR}/backends)

set(IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/imgui_demo.cpp

    ${IMGUI_BACKENDS}/imgui_impl_sdl2.cpp
    ${IMGUI_BACKENDS}/imgui_impl_sdlrenderer2.cpp
)

include_directories(
    include
    ${SDL2_INCLUDE_DIRS}
    ${IMGUI_DIR}
    ${IMGUI_BACKENDS}
)

# Your sources
file(GLOB SRC_FILES
    src/*.cpp
    src/*.cu
)

add_executable(openlife ${SRC_FILES} ${IMGUI_SOURCES})

target_link_libraries(openlife ${SDL2_LIBRARIES} SDL2main SDL2)

set_target_properties(openlife PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
