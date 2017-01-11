#ifndef YGZ_MATCHER_H_
#define YGZ_MATCHER_H_

#include "ygz/Basic/Common.h"

namespace ygz {

struct Frame;    
struct MapPoint;    
struct Feature;    

// 匹配相关的算法，包括特征点的匹配法和直接法的匹配法
class Matcher 
{
public:
    
    // 特征匹配的参数，从orb-slam2中拷贝
    struct Options {
        int th_high = 100;       // 这两个在搜索匹配时用
        int th_low = 50;         // 低阈值
        float knnRatio = 0.8;    // knn 时的比例
        
        bool checkOrientation = false;  // 除了检测描述之外是否检查旋转
        
        float initMatchRatio = 3.0;     // 初始化时的比例
        int init_low = 30;              // 这两个在初始化时用于检测光流结果是否正确
        int init_high = 80;             
        
        
    } _options;
    
        
    static const int HISTO_LENGTH = 30;  // 旋转直方图的size
    
    Matcher();
    
    // 特征点法的匹配
    // Computes the Hamming distance between two ORB descriptors
    // 两个描述子之间的Hamming距离，它们必须是1x32的ORB描述
    static int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);
    
    // 搜索特征点以建立三角化结果
    // 给定两帧之间的 Essential，计算 matched points 
    // 结果中的 kf1 关键点状态必须没有三角化，kf2中则无所谓
    int SearchForTriangulation( 
        Frame* kf1, Frame* kf2, const Matrix3d& E12, 
        vector< pair<int, int> >& matched_points, 
        const bool& onlyStereo = false
    );
    
    // 在Keyframe之间搜索匹配情况，利用BoW加速
    int SearchByBoW( Frame * kf1, Frame* kf2, map<int, int>& matches );
    
    // 在KeyFrame和地图之间搜索匹配情况
    // int SearchByProjection(Frame *F, const std::vector<MapPoint*> &vpMapPoints, const float th=3);
    
    // 计算两个帧之间的特征描述是否一致
    // 这是在初始化里用的。初始化使用了光流跟踪了一些点，但我们没法保证跟踪成功，所以需要再检查一遍它们的描述量
    // 第三个参数内指定了光流追踪的match，如果描述不符合，就会从里面剔除
    int CheckFrameDescriptors( Frame* frame1, Frame* frame2, list<pair<int,int>>& matches ); 
    
    
    // ****************************************************************************************************
    // 直接法的匹配 
    
    // 用直接法判断能否从在当前图像上找到某地图点的投影
    bool FindDirectProjection( Frame* ref, Frame* curr, MapPoint* mp, Vector2d& px_curr );
    
    // model based sparse image alignment
    // 通过参照帧中观测到的3D点，预测当前帧的pose，稀疏直接法
    bool SparseImageAlignment( Frame* ref, Frame* current );
    
private:
    // 内部函数
    // 计算旋转直方图中三个最大值
    void ComputeThreeMaxima( vector<int>* histo, const int L, int& ind1, int& ind2, int& ind3 );
    
    bool CheckDistEpipolarLine( const Vector3d& pt1, const Vector3d& pt2, const Eigen::Matrix3d& E12 );
    
    // 对每层金字塔计算的 image alignment 
    bool SparseImageAlignmentInPyramid( Frame* ref, Frame* current, int pyramid );
    
    
    void GetWarpAffineMatrix (
        const Frame* ref,
        const Frame* curr,
        const Vector2d& px_ref,
        const Vector3d& pt_ref,
        const int & level,
        const SE3& TCR,
        Eigen::Matrix2d& ACR
    );

    // perform affine warp
    void WarpAffine (
        const Eigen::Matrix2d& ACR,
        const cv::Mat& img_ref,
        const Vector2d& px_ref,
        const int& level_ref,
        const int& search_level,
        const int& half_patch_size,
        uint8_t* patch
    );

    // 计算最好的金字塔层数
    // 选择一个分辨率，使得warp不要太大
    inline int GetBestSearchLevel (
        const Eigen::Matrix2d& ACR,
        const int& max_level )
    {
        int search_level = 0;
        double D = ACR.determinant();
        while ( D > 3.0 && search_level < max_level ) {
            search_level += 1;
            D *= 0.25;
        }
        return search_level;
    }
    
    // 计算参照帧中的图像块
    void PrecomputeReferencePatches( Frame* ref, int level ); 
    
    // 匹配局部地图用的 patch, 默认8x8
    uchar _patch[WarpPatchSize*WarpPatchSize];
    // 带边界的，左右各1个像素
    uchar _patch_with_border[(WarpPatchSize+2)*(WarpPatchSize+2)];
    
    vector<uchar*> _patches_align;      // 等待推倒的patches

};
    
}

#endif