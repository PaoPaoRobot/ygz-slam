cmake_minimum_required( VERSION 2.8 )
project ( ygz )

# compiler
set( CMAKE_CXX_COMPILER "g++" )
#set( CMAKE_CXX_COMPILER "clang++" )

# Release config
set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -march=native -O3 " )

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

# TODO: add gtsam and other libs you will use

set( THIRD_PARTY_LIBS 
    ${OpenCV_LIBS}
    ${Sophus_LIBRARIES}
    ${GLOG_LIBRARY}
    ${Boost_LIBRARIES}
)

############### executable ######################
add_subdirectory( src )
add_subdirectory( test )
add_subdirectory( examples )
