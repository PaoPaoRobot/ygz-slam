#ifndef YGZ_INITIALIZER_H_
#define YGZ_INITIALIZER_H_

#include "ygz/Basic.h"

namespace ygz 
{
    
// 单目初始化
// 根据特征点跟踪的结果，分解E或H获得初始化结果
// 初始化只需完成位姿估计，给出 inlier 即可，三角化由其他算法完成
// 参照ORB-SLAM2设计，同时考虑H和E两种情况
    
class Initializer 
{
public:
    // 尝试进行初始化
    // 传入已经匹配好的两组点，以及参考帧与当前帧的指针
    // 允许有一定程度的误匹配，但最好在调用之前已经检查匹配是否合理，并且有一定程度的视差
    // 成功初始化时，返回true
    bool TryInitialize(
        const vector<Vector2d>& px1, 
        const vector<Vector2d>& px2, 
        Frame* ref, 
        Frame* curr
    ); 
    
    struct Option 
    {
        float _sigma = 1.0;
        float _sigma2 = 1.0;
        int _max_iter = 200;
    } _options;
    
private:
    void FindHomography(vector<bool> &vbMatchesInliers, float &score, Matrix3d& H21);
    // 假设场景为非平面情况下通过前两帧求取Fundamental矩阵(current frame 2 到 reference frame 1),并得到该模型的评分
    void FindFundamental(vector<bool> &vbInliers, float &score, cv::Mat &F21);

    // 被FindHomography函数调用具体来算Homography矩阵
    cv::Mat ComputeH21(const vector<Vector2d> &vP1, const vector<Vector2d> &vP2);
    // 被FindFundamental函数调用具体来算Fundamental矩阵
    cv::Mat ComputeF21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2);

    // 被FindHomography函数调用，具体来算假设使用Homography模型的得分
    float CheckHomography(const cv::Mat &H21, const cv::Mat &H12, vector<bool> &vbMatchesInliers, float sigma);
    // 被FindFundamental函数调用，具体来算假设使用Fundamental模型的得分
    float CheckFundamental(const cv::Mat &F21, vector<bool> &vbMatchesInliers, float sigma);

    // 分解F矩阵，并从分解后的多个解中找出合适的R，t
    bool ReconstructF(vector<bool> &vbMatchesInliers, cv::Mat &F21, cv::Mat &K,
                      cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated);

    // 分解H矩阵，并从分解后的多个解中找出合适的R，t
    bool ReconstructH(vector<bool> &vbMatchesInliers, cv::Mat &H21, cv::Mat &K,
                      cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated);

    // 通过三角化方法，利用反投影矩阵将特征点恢复为3D点
    void Triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D);

    // 归一化三维空间点和帧间位移t
    void Normalize(
        const vector<Vector2d>& pixels, 
        vector<Vector2d> &pixels_normalized,
        Matrix3d &T
    );

    // ReconstructF调用该函数进行cheirality check，从而进一步找出F分解后最合适的解
    int CheckRT(const cv::Mat &R, const cv::Mat &t, const vector<cv::KeyPoint> &vKeys1, const vector<cv::KeyPoint> &vKeys2,
                       const vector<Match> &vMatches12, vector<bool> &vbInliers,
                       const cv::Mat &K, vector<cv::Point3f> &vP3D, float th2, vector<bool> &vbGood, float &parallax);

    // F矩阵通过结合内参可以得到Essential矩阵，该函数用于分解E矩阵，将得到4组解
    void DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t);
    
    vector<bool> _inliers;      // 判断匹配点对是否为inlier
    vector<vector<size_t>> _set;
    
    SE3 _T21;                   // 估得的 T21
    
    vector<Vector2d>* _px1=nullptr;
    vector<Vector2d>* _px2=nullptr;
    int _num_points=0   ;            // 总共匹配点的数量
    Frame* _ref=nullptr, _curr=nullptr;
};

}


#endif // YGZ_INITIALIZER_H_