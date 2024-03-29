cmake_minimum_required(VERSION 2.8)

project (ublox_linux)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/lib)

add_definitions(-std=c++11 -Wno-deprecated -fconcepts -Wall -DLINUX_PLATFORM)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

link_directories(
    /usr/local/lib
    /usr/lib
    ${CMAKE_HOME_DIRECTORY}/lib)

include_directories(
    ${CMAKE_HOME_DIRECTORY}/../inc
    ${CMAKE_HOME_DIRECTORY}/../SparkFun_u-blox_GNSS_Arduino_Library/src)

set(ublox_parser_src
    ../SparkFun_u-blox_GNSS_Arduino_Library/src/SparkFun_u-blox_GNSS_Arduino_Library.cpp
    ../src/Print.cpp
    ../src/Utils.cpp
    )

add_library(
    ublox_linux SHARED ${ublox_parser_src})

#
# Ublox f9p Test
#

set(ublox_f9p_test_src ../linux_examples/ublox_f9p_test.cpp)

add_executable(ublox_f9p_test ${ublox_f9p_test_src})

target_link_libraries(ublox_f9p_test ublox_linux util)

#
# Ublox f9p i2c Test
#

set(ublox_f9p_i2c_test_src ../linux_examples/ublox_f9p_i2c_test.cpp)

add_executable(ublox_f9p_i2c_test ${ublox_f9p_i2c_test_src})

target_link_libraries(ublox_f9p_i2c_test ublox_linux util)

#-----------RTK Capture--------------

set(rtk_capture_src ../linux_examples/rtk_capture.cpp)

add_executable(rtk_capture ${rtk_capture_src})

target_link_libraries(rtk_capture ublox_linux util ${OpenCV_LIBS} pthread)