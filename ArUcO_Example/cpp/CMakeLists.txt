cmake_minimum_required(VERSION 3.2)
project(pose_example)

set (CMAKE_CXX_STANDARD 11)
find_package(OpenCV REQUIRED)

include_directories(${OPENCV_INCLUDE_DIRS})

link_directories(${OpenCV_LIBRARY_DIRS})

add_executable(pose_estimation main.cpp)

target_link_libraries(pose_estimation ${OpenCV_LIBRARIES})

target_compile_options(pose_estimation PRIVATE -O3 -std=c++11)

