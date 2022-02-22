cmake_minimum_required( VERSION 3.2 )

set (CMAKE_CXX_STANDARD 17)

# Create Project
project( Multi_PCD_Merger )
add_executable( Multi_PCD_Merger main.cpp )

# Set StartUp Project (Option)
# (This setting is able to enable by using CMake 3.6.0 RC1 or later.)
set_property( DIRECTORY PROPERTY VS_STARTUP_PROJECT "project" )

# Find Packages
find_package( PCL 1.8 REQUIRED )

if( PCL_FOUND )
    # [C/C++]>[General]>[Additional Include Directories]
    include_directories( ${PCL_INCLUDE_DIRS} )

    # [C/C++]>[Preprocessor]>[Preprocessor Definitions]
    add_definitions( ${PCL_DEFINITIONS} )

    # For Use Not PreCompiled Features
    #add_definitions( -DPCL_NO_PRECOMPILE )

    # [Linker]>[General]>[Additional Library Directories]
    link_directories( ${PCL_LIBRARY_DIRS} )

    # [Linker]>[Input]>[Additional Dependencies]
    target_link_libraries( Multi_PCD_Merger ${PCL_LIBRARIES} )
endif()