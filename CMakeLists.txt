cmake_minimum_required( VERSION 2.8 )
project ( ygz )

# compiler
set( CMAKE_CXX_COMPILER "g++" )
#set( CMAKE_CXX_COMPILER "clang++" )
# debug 
set( CMAKE_BUILD_TYPE "Debug" )
set( CMAKE_CXX_FLAGS "-std=c++11 -march=native -Wno-reorder" )
add_definitions( "-DDEBUG_VIZ" )

# Release config
# set( CMAKE_BUILD_TYPE "Release" )
# set( CMAKE_CXX_FLAGS "-std=c++11 -march=native -O3 " )

# Set build flags, set ARM_ARCHITECTURE environment variable on Odroid
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -D_LINUX -D_REENTRANT -Wno-unused-variable -Wno-unused-but-set-variable -Wno-unknown-pragmas")
IF(DEFINED ENV{ARM_ARCHITECTURE})
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon -march=armv7-a")
ELSE()
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mmmx -msse -msse -msse2 -msse3 -mssse3")
ENDIF()

# Debug config 
# set( CMAKE_BUILD_TYPE "Debug" )
# set( CMAKE_CXX_FLAGS "-std=c++11 -march=native" )

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )
set( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin )
set( LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib )
include_directories( ${PROJECT_SOURCE_DIR}/include )

############### dependencies ######################
# Eigen
include_directories( "/usr/include/eigen3" )
# OpenCV
find_package( OpenCV 3.1 REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
# Sophus 
find_package( Sophus REQUIRED )
include_directories( ${Sophus_INCLUDE_DIRS} )
# glog 
find_package( Glog REQUIRED )
include_directories( ${GLOG_INCLUDE_DIRS} )
# boost
find_package( Boost COMPONENTS filesystem REQUIRED )
include_directories( ${Boost_INCLUDE_DIRS} )
# fast 
find_package( fast REQUIRED )
include_directories( ${fast_INCLUDE_DIRS} )
# g2o 
find_package( G2O REQUIRED )
include_directories( ${G2O_INCLUDE_DIRS} )
# csparse, needed in g2o 
find_package( CSparse REQUIRED )
include_directories( ${CSPARSE_INCLUDE_DIR} )
# ceres 
find_package( Ceres REQUIRED )
include_directories( ${CERES_INCLUDE_DIRS} )
# pangolin 
find_package( Pangolin REQUIRED )
include_directories( ${Pangolin_INCLUDE_DIRS} )



# TODO: add gtsam and other libs you will use

set( THIRD_PARTY_LIBS 
    ${OpenCV_LIBS}
    ${Sophus_LIBRARIES}
    ${GLOG_LIBRARY}
    ${Boost_LIBRARIES}
    ${fast_LIBRARIES}
    
    # g2o has a group of mess things 
    g2o_core g2o_stuff g2o_csparse_extension g2o_types_sba g2o_solver_csparse 
    ${CSPARSE_LIBRARY}
    ${CERES_LIBRARIES}
    ${Pangolin_LIBRARIES}
)

############### executable ######################
add_subdirectory( src )
add_subdirectory( test )
add_subdirectory( examples )
