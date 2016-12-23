#ifndef YGZ_ORB_MATCHER_H_
#define YGZ_ORB_MATCHER_H_

#include "ygz/common_include.h" 

namespace ygz {
    
class Frame;
    
class ORBMatcher 
{
public:
    // Constructor 
    ORBmatcher(float nnratio=0.6, bool checkOri=true);
    
    // Computes the Hamming distance between two ORB descriptors
    static int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);
    
    // 在Keyframe之间搜索匹配情况
    int SearchByBoW( Frame * kf1, Frame* kf2 );
    
    // 在KeyFrame和地图之间搜索匹配情况
    int SearchByProjection(Frame &F, const std::vector<MapPoint*> &vpMapPoints, const float th=3);
    
private:
    float _nnRatio;
    bool _checkOrientation;
};
    
}