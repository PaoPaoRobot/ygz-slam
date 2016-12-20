#ifndef YGZ_ORB_EXTRACTOR_H
#define YGZ_ORB_EXTRACTOR_H

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "ygz/common_include.h"



namespace ygz {
    
    
const int PATCH_SIZE = 31;
const int HALF_PATCH_SIZE = 15;
const int EDGE_THRESHOLD = 19;

float IC_Angle(
    const Mat& image, cv::Point2f pt,  
    const vector<int> & u_max
);
    
class ORBExtractor {
    
public:
    ORBExtractor();
    
    // 根据 直接法 提取的角点，将它升级为特征点，并计算描述子
    // 批量计算 
    void Compute( const cv::Mat& image, const vector<Vector2d>& corners, vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors );
    // 单个儿计算 
    void Compute( const cv::Mat& image, const Vector2d& corners, cv::KeyPoint& keypoints, cv::Mat& descriptors );
    std::vector<int> umax;
    
protected:
    cv::Ptr<cv::ORB> _orb ;
};

}

#endif 