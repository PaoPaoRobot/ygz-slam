#ifndef COMMON_INCLUDE_H_
#define COMMON_INCLUDE_H_

// std 
#include <vector>
#include <list>
#include <memory>
#include <string>
#include <iostream>
#include <set>
#include <unordered_map>
#include <map>
#include <algorithm>
#include <mutex>
#include <thread>

using namespace std; 

// define the commonly included file to avoid a long include list
// for Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::Matrix2f;
using Eigen::Matrix3f;
using Eigen::Matrix4f;
using Eigen::Matrix2d;
using Eigen::Matrix3d;
using Eigen::Matrix4d;
// other things I need in optimiztion 
typedef Eigen::Matrix<double, 6, 1> Vector6d; 

#include <Eigen/StdVector> // for vector of Eigen objects 
typedef vector<Vector3d, Eigen::aligned_allocator<Vector3d>> VecVector3d;

// for Sophus
#include <sophus/se3.h>
#include <sophus/so3.h>
using Sophus::SO3;
using Sophus::SE3;

// for cv
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using cv::Mat;

// for glog
#include <glog/logging.h>


// ceres
#include <ceres/ceres.h>

// for DBoW3 
#include <BowVector.h>
#include <FeatureVector.h>
#include "Vocabulary.h"
typedef DBoW3::Vocabulary ORBVocabulary;


// ********************************************************************************
// 常量定义

// 稀疏直接法里用的pattern
enum {PATTERN_SIZE = 8};
const double PATTERN_DX[PATTERN_SIZE] = {0,-1,1,1,-2,0,2,0};
const double PATTERN_DY[PATTERN_SIZE] = {0,-1,-1,1,0,-2,0,2};

// Local mapping中使用的patch大小
const int WarpHalfPatchSize = 4;    
const int WarpPatchSize = 8;

#endif
