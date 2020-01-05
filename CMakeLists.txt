CMAKE_MINIMUM_REQUIRED(VERSION 3.8 FATAL_ERROR)

PROJECT(Sfs-Sr-Single LANGUAGES CXX CUDA)

# ------------------------------------------------------------------------
# Build Tests
OPTION(BUILD_TESTS "Build tests" OFF)

# ------------------------------------------------------------------------
# CMake configuration
IF(NOT CMAKE_BUILD_TYPE)
	# CMAKE_BUILD_TYPE: RelWithDebInfo, Release or Debug
    SET(CMAKE_BUILD_TYPE Debug)
ENDIF()

# CMake paths
# Externals directories (from environment variable "EXT")
LIST(APPEND CMAKE_PREFIX_PATH "$ENV{EXT}")
# include files in cmake subfolder
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# executable and library output paths
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

# ------------------------------------------------------------------------

OPTION(BUILD_APPS "Build MV-Depth-SFS applications" ON)


# ------------------------------------------------------------------------
# dependencies

FIND_PACKAGE(CUDA REQUIRED)
INCLUDE_DIRECTORIES(${CUDA_TOOLKIT_INCLUDE})
SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++11")
SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")


# OpenCV
FIND_PACKAGE(OpenCV 3 REQUIRED core highgui imgproc)

INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/third_party/Eigen/include)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/third_party/sophus/include)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/third_party/mLib/include)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/third_party/Opt/API/release/include)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/third_party/terra/include)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/third_party/CLI/include)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/third_party/Pangolin/include)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/third_party/cuda-image/include)


LINK_DIRECTORIES(${PROJECT_SOURCE_DIR}/third_party/Opt/API/release/lib)
SET(Opt_LIBRARIES
        Opt
        OptDev
)

# Terra
LINK_DIRECTORIES(${PROJECT_SOURCE_DIR}/third_party/terra/lib)
SET(Terra_LIBRARIES terra)

# Pangolin
LINK_DIRECTORIES(${PROJECT_SOURCE_DIR}/third_party/Pangolin/lib)

# CudaImage
LINK_DIRECTORIES(${PROJECT_SOURCE_DIR}/third_party/cuda-image/lib)

# GTest
IF(BUILD_TESTS)
    FIND_PACKAGE(GTest)
    INCLUDE_DIRECTORIES(${GTEST_INCLUDE_DIR})
ENDIF()

# ------------------------------------------------------------------------
# C++ flags

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -funroll-loops")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

ADD_DEFINITIONS(
    -DSOURCE_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"
)

# ------------------------------------------------------------------------
# CUDA flags
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -use_fast_math)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
	SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O0 -g -G)
elseif (CMAKE_BUILD_TYPE STREQUAL "Release")
	SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3)
elseif (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
	SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3 -g -G)
endif (CMAKE_BUILD_TYPE STREQUAL "Debug")

# ------------------------------------------------------------------------
# modules

LINK_DIRECTORIES(${LIBRARY_OUTPUT_PATH})

# core library
LIST(APPEND MODULES "libcore")

# image library
LIST(APPEND MODULES "libimage")

# solver library
LIST(APPEND MODULES "libsolver")

# data library
LIST(APPEND MODULES "libdata")

FOREACH(MODULE ${MODULES})
    ADD_SUBDIRECTORY(${MODULE})
    INCLUDE_DIRECTORIES(${MODULE}/include)
ENDFOREACH()

# ------------------------------------------------------------------------
# applications

LIST(APPEND APPS "sfs")


FOREACH(APP ${APPS})
    ADD_SUBDIRECTORY(apps/${APP})
    INCLUDE_DIRECTORIES(${APP}/include)
ENDFOREACH()
