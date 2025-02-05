cmake_minimum_required(VERSION 3.10)

# Project name and version
project(RayTracer VERSION 1.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# Enable compiler warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -O2")
include_directories(${CMAKE_SOURCE_DIR}/include)
# Add source files (use globbing to include all .cpp files)
file(GLOB SOURCES "src/*.cpp")

# Create the executable target
add_executable(raytracer ${SOURCES})

# Specify additional include directories (if any)
target_include_directories(raytracer PRIVATE ${CMAKE_SOURCE_DIR}/include)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
endif()
