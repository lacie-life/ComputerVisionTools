cmake_minimum_required(VERSION 3.0.2)
project(realsene_capture)

# Enable C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

find_package(realsense2 REQUIRED)
find_package(OpenCV 4.2 REQUIRED)

# Add the application sources to the target
add_executable(${PROJECT_NAME} capture.cpp)

# Link librealsense2 to the target
target_link_libraries(${PROJECT_NAME} ${realsense2_LIBRARY} ${OpenCV_LIBS})

