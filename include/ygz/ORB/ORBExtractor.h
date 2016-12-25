#ifndef YGZ_ORB_EXTRACTOR_H
#define YGZ_ORB_EXTRACTOR_H

#include "ygz/common_include.h"

namespace ygz {
    
class Frame;
    
class ORBExtractor {
    
    // ORB 相关常量
    const int PATCH_SIZE = 31;
    const int HALF_PATCH_SIZE = 15;
    const int EDGE_THRESHOLD = 19;
    
public:
    ORBExtractor();
    
    // 根据 直接法 提取的角点，将它升级为特征点，并计算描述子
    // 主要是对旋转的计算，因为直接法中的角点即是FAST
    // 批量计算 
    // void Compute( const cv::Mat& image, const vector<Vector2d>& corners, vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors );
    
    // 单个儿计算 
    // void Compute( const cv::Mat& image, const Vector2d& corners, cv::KeyPoint& keypoints, cv::Mat& descriptors );
    
    // 针对 Frame 的调用，计算Frame中每个特征点的旋转＋描述
    void Compute( Frame* frame );
    
    // 计算角度
    float IC_Angle(
        const Mat& image, cv::Point2f pt,  
        const vector<int> & u_max
    );
    
    void ComputeOrbDescriptor(
        const cv::KeyPoint& kpt,
        const Mat& img, 
        const cv::Point* pattern,
        uchar* desc
    );
    
protected:
    std::vector<int> _umax;
    // cv::Ptr<cv::ORB> _orb ;
    vector<cv::Point> pattern;
};

}

#endif 