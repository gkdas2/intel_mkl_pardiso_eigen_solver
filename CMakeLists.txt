cmake_minimum_required(VERSION 3.25)
project(pardiso_solver C CXX)

set(CMAKE_BUILD_TYPE Release)
set(MKL_INCLUDE_DIRS "/opt/intel/oneapi/mkl/latest/include" )
set(MKL_LIBRARIES "/opt/intel/oneapi/mkl/latest/lib/intel64" )


# Set compiler flags
add_compile_options(-O3 -m64 -fopenmp)
# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)


#include MKL include headers
include_directories(${MKL_INCLUDE_DIRS})

# Find Eigen
find_package (Eigen3 3.3 REQUIRED NO_MODULE)


# Add source codes
set (source_dir "${PROJECT_SOURCE_DIR}/pardiso")
file (GLOB source_files "${source_dir}/*.cpp")

# Add executable
add_executable(pardiso_solver ${source_files})

# Specify the language for source files
set_source_files_properties(${source_files} PROPERTIES LANGUAGE CXX)

# Link with MKL and eigen libraries. the intel is linked statically. see compiler flags online if dynamic is desired. make sure to verify results in that case 
TARGET_LINK_LIBRARIES(pardiso_solver PUBLIC Eigen3::Eigen -L$(MKL_LIBRARIES) "-Wl,--start-group" -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core "-Wl,--end-group -lgomp -lpthread -lm -ldl")
