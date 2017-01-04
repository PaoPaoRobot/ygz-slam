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
        int th_high = 100;       // 这两个在搜索匹配时用
        int th_low = 50;         // 低阈值
        float knnRatio = 0.8;
        bool checkOrientation = false;
        float initMatchRatio = 3.0; 
        int init_low = 30;
        int init_high = 80;
    } _options;
    
    static const int HISTO_LENGTH = 30;
    
    // Constructor 
    ORBMatcher();
    
    // Computes the Hamming distance between two ORB descriptors
    static int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);
    
    // 匹配特征点以建立三角化结果
    // 给定两帧之间的Fundamental，计算 matched points 
    int SearchForTriangulation( 
        Frame* kf1, Frame* kf2, const Matrix3d& E12, 
        vector< pair<int, int> >& matched_points, 
        const bool& onlyStereo = false
    );
    
    // 在Keyframe之间搜索匹配情况
    int SearchByBoW( Frame * kf1, Frame* kf2, map<int, int>& matches );
    
    // 在KeyFrame和地图之间搜索匹配情况
    // int SearchByProjection(Frame *F, const std::vector<MapPoint*> &vpMapPoints, const float th=3);
    
    // 计算两个帧之间的特征描述是否一致
    // 注意：这里需要两个frame已经提好了一对一的特征
    int CheckFrameDescriptors( Frame* frame1, Frame* frame2, vector<pair<int,int>>& matches ); 
    
    
private:
    // 计算旋转直方图中三个最大值
    void ComputeThreeMaxima( vector<int>* histo, const int L, int& ind1, int& ind2, int& ind3 );
    
    bool CheckDistEpipolarLine( const cv::KeyPoint& kp1, const cv::KeyPoint& kp2, const Eigen::Matrix3d& F12 );
};
    
}

#endif