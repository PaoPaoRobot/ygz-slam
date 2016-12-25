#pragma once 
#ifndef YGZ_ORB_MATCHER_H_
#define YGZ_ORB_MATCHER_H_

#include "ygz/common_include.h" 

namespace ygz {
    
class Frame;
    
class ORBMatcher 
{
public:
    
    // 特征匹配的参数，从orb-slam2中拷贝
    struct Options {
        int th_high = 100;
        int th_low = 30;
        int histo_length = 30;
        float ratio = 3.5;
    } _options;
    
    // Constructor 
    ORBMatcher(float nnratio=0.6, bool checkOri=true);
    
    // Computes the Hamming distance between two ORB descriptors
    static int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);
    
    // 在Keyframe之间搜索匹配情况
    // int SearchByBoW( Frame * kf1, Frame* kf2 );
    
    // 在KeyFrame和地图之间搜索匹配情况
    // int SearchByProjection(Frame *F, const std::vector<MapPoint*> &vpMapPoints, const float th=3);
    
    // 计算两个帧之间的特征描述是否一致
    bool CheckFrameDescriptors( Frame* frame1, Frame* frame2, vector<bool>& inliers ); 
    
    
private:
    float _nnRatio;
    bool _checkOrientation;
};
    
}

#endif