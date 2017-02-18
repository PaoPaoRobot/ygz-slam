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
    // 请注意生成的3D点要单独获取，这里并不会自动向地图中添加地图点
    bool TryInitialize(
        vector<Vector2d>& px1, 
        vector<Vector2d>& px2, 
        Frame* ref, 
        Frame* curr
    ); 
    
    // 获得估计的解
    SE3 GetT21() const 
    {
        return _T21;
    }
    
    // 获得3D点和内点
    void GetTriangluatedPoints( vector<Vector3d>& pts_3d, vector<bool>& inliers )
    {
        pts_3d = _pts_triangulated;
        inliers = _inliers;
    }
    
    // 配置参数
    struct Option 
    {
        float _sigma = 2.0;     // 误差方差
        float _sigma2 = 4.0;    // 误差方差平方
        int _max_iter = 200;    // 最大迭代次数
        double _min_parallex =1.0;         // 最小平行夹角
        int _min_triangulated_pts=8;      // 最少被三角化点的数量
        double good_point_ratio_H = 0.9;   // 在分解H时，好点占总点数的最小比例
    } _options;
    
private:
    void FindHomography(vector<bool> &vbMatchesInliers, float &score, Matrix3d& H21);
    // 假设场景为非平面情况下通过前两帧求取Fundamental矩阵(current frame 2 到 reference frame 1),并得到该模型的评分
    void FindFundamental(vector<bool> &vbInliers, float &score, Matrix3d &F21);

    // 被FindHomography函数调用具体来算Homography矩阵
    Matrix3d ComputeH21(const vector<Vector2d> &vP1, const vector<Vector2d> &vP2);
    // 被FindFundamental函数调用具体来算Fundamental矩阵
    Matrix3d ComputeF21(const vector<Vector2d> &vP1, const vector<Vector2d> &vP2);

    // 被FindHomography函数调用，具体来算假设使用Homography模型的得分
    float CheckHomography(const Matrix3d &H21, const Matrix3d &H12, vector<bool> &inliers, float sigma);
    // 被FindFundamental函数调用，具体来算假设使用Fundamental模型的得分
    float CheckFundamental(const Matrix3d &F21, vector<bool> &vbMatchesInliers, float sigma);

    // 分解F矩阵，并从分解后的多个解中找出合适的R，t
    bool ReconstructF( vector<bool> &inliers, Matrix3d &F21, Matrix3d &K,
                      Matrix3d &R21, Vector3d &t21, vector<Vector3d> &vP3D, 
                      vector<bool> &vbTriangulated, float minParallax, int minTriangulated);

    // 分解H矩阵，并从分解后的多个解中找出合适的R，t
    bool ReconstructH( vector<bool> &inliers, Matrix3d &H21, Matrix3d &K,
                      Matrix3d &R21, Vector3d &t21, vector<Vector3d> &vP3D, 
                      vector<bool> &triangulated, float minParallax, int minTriangulated );

    // 通过三角化方法，利用反投影矩阵将特征点恢复为3D点
    void Triangulate(
        const Vector2d &kp1, const Vector2d &kp2,       // 2D 点
        const Eigen::Matrix<double,3,4> &P1,            // 第1个相机投影矩阵
        const Eigen::Matrix<double,3,4> &P2,            // 第2个相机投影矩阵
        Vector3d &x3D                                   // 3D 点
    );

    // 归一化三维空间点和帧间位移t
    void Normalize(
        const vector<Vector2d>& pixels, 
        vector<Vector2d> &pixels_normalized,
        Matrix3d &T
    );

    // ReconstructH调用该函数进行cheirality check，从而进一步找出H分解后最合适的解
    // 返回好的点的个数
    /**
     * @brief check whether the estimated R,t is valid
     * @param[in] R rotation matrix
     * @param[in] t translation vector
     * @param[in] inliers inliers given by CheckHomography or CheckFundamental
     * @param[in] K camera intrinsics
     * @param[out] p3D triangulated 3D points 
     * @param[out] th2 reprojection ch2 error threshold 
     * @param[out] good good points in triangulation
     * @param[out] parallax max parallax angle
     * @param[in] check_reprojection whether need to check the reprojection error
     * @return number of good points 
     */
    int CheckRT( const Matrix3d &R, const Vector3d &t,  // H分解得到的R,t 
                vector<bool> &inliers,                  // 每个点是否为inliers
                const Matrix3d& K,                      
                vector<Vector3d> &p3D,                  
                float th2, 
                vector<bool> &good, 
                double &parallax,
                bool check_reprojection = true          // C++ 必须要把带默认值的参数搁后面。。。
    );

    // F矩阵通过结合内参可以得到Essential矩阵，该函数用于分解E矩阵，将得到4组解
    void DecomposeE(const Matrix3d &E, Matrix3d &R1, Matrix3d &R2, Vector3d &t);
    
    vector<bool> _inliers;      // 判断匹配点对是否为inlier
    vector<Vector3d> _pts_triangulated; // 三角化的点
    SE3 _T21;                   // 估得的 T21
    
    vector<vector<size_t>> _set;        
    vector<Vector2d> _px1;
    vector<Vector2d> _px2;
    int _num_points=0;            // 总共匹配点的数量
    
    Frame* _ref=nullptr; 
    Frame* _curr=nullptr;
    
    // 分解H得到的结果
    struct HomographyDecomposition
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        Eigen::Vector3d t;
        Eigen::Matrix3d R;
        Eigen::Vector3d n;
    };
};

}


#endif // YGZ_INITIALIZER_H_