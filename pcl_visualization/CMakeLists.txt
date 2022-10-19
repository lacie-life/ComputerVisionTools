cmake_minimum_required(VERSION 3.23)
project(pcl_visualization_test)

set(CMAKE_CXX_STANDARD 14)
find_package(PCL 1.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (pcl_visualization_test main.cpp)
target_link_libraries (pcl_visualization_test ${PCL_LIBRARIES})
