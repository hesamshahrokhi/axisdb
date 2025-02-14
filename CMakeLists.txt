cmake_minimum_required(VERSION 3.10)
project(AxisDB LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable warnings and optimization
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3")

# Find HDF5 package
find_package(HDF5 REQUIRED COMPONENTS CXX)

# Add executable
add_executable(main src/main.cpp src/lib.cpp)

# Include HDF5 directories
target_include_directories(main PRIVATE ${HDF5_INCLUDE_DIRS})

# Link HDF5 libraries
target_link_libraries(main PRIVATE ${HDF5_CXX_LIBRARIES} ${HDF5_LIBRARIES})
