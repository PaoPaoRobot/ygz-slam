#ifndef INITIALIZER_H_
#define INITIALIZER_H_

#include "ygz/common_include.h"
#include "ygz/frame.h"

namespace ygz {

/**************************************
 * The initialize class
 * it is stored in visual odometry class, used when initialing
 * note that in RGB-D or stereo mode, we do not need this step.
 * ***********************************/

// 初始化
// 根据特征点跟踪的结果，分解E或H获得初始化结果
class Initializer {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    struct HomographyDecomposition
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        Eigen::Vector3d t;
        Eigen::Matrix3d R;
        Eigen::Vector3d n;
        double   d;

        // Resolved  Composition
        SE3 T;
        int score;
    };

    typedef vector<HomographyDecomposition, Eigen::aligned_allocator<HomographyDecomposition>> HomographyDecs;
    
    Initializer();

    // try to initialize, given two sets of pixels, return the recovered motion and structure
    // NOTE px1 and px2 are pixel coordinates, and pt1, pt2 are normalized coordinates
    bool TryInitialize(
        const vector<Vector2d>& px1,
        const vector<Vector2d>& px2,
        Frame* ref,
        Frame* curr
    );

    // 测试是否可以初始化，期望平均视差大于阈值
    inline bool Ready( const float& mean_disparity ) {
        LOG(INFO) << "checking disparity "<<mean_disparity<<endl;
        return mean_disparity > _min_disparity;
    }

    // 获得初始化的内点
    vector<bool> GetInliers() const {
        return _inliers;
    }

protected:
    // check if homography works
    bool TestHomography();

    // check if fundamental works
    bool TestEssential();

    // decompose H
    HomographyDecs DecomposeHomography();
    
    // find best decompostion 
    void FindBestDecomposition( HomographyDecs& decomp );

    double SampSonusError( 
        const cv::Point2f& v2Dash, 
        const Eigen::Matrix3d& Essential, 
        const cv::Point2f& v2 
    );
    
    // triagulate the points between ref and curr 
    void Triangulate( const SE3& T21, vector<Vector3d>& pts_triangulated );
    
    // rescale the map, return the map 
    double RescaleMap( vector<Vector3d>&pts );
    
private:
    // params
    int _max_iterations;
    float _min_disparity;
    int _min_inliers;
    float _pose_init_thresh;

    // data
    Frame* _frame;
    vector<cv::Point2f> _px1, _px2;
    vector<cv::Point2f> _pt1, _pt2;
    Mat _H_estimated, _E_estimated;
    SE3 _T21;

    vector<bool> _inliers;
};

}

#endif
