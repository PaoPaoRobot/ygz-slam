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
struct HomographyDecomposition
{
    Eigen::Vector3d t;
    Eigen::Matrix3d R;
    Eigen::Vector3d n;
    double   d;

    // Resolved  Composition
    SE3 T;
    int score;
};

class Initializer {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Initializer();

    // try to initialize, given two sets of pixels, return the recovered motion and structure
    // NOTE px1 and px2 are pixel coordinates, and pt1, pt2 are normalized coordinates
    bool TryInitialize(
        const vector<Vector2d>& px1,
        const vector<Vector2d>& px2,
        const Frame::Ptr& ref,
        Frame::Ptr curr
    );

    inline bool Ready( const float& mean_disparity ) {
        LOG(INFO) << "checking disparity "<<mean_disparity<<endl;
        return mean_disparity > _min_disparity;
    }

    vector<bool> GetInliers() const {
        return _inliers;
    }

protected:
    // check if homography works
    bool TestHomography();

    // check if fundamental works
    bool TestEssential();

    // decompose H
    vector<HomographyDecomposition> DecomposeHomography();
    
    // find best decompostion 
    void FindBestDecomposition( vector<HomographyDecomposition>& decomp );

    double SampSonusError( 
        const cv::Point2f& v2Dash, 
        const Eigen::Matrix3d& Essential, 
        const cv::Point2f& v2 
    );
    
    // triagulate the points between ref and curr 
    void Triangulate( const SE3& T21, vector<Vector3d>& pts_triangulated );
    
    // rescale the map, return the map 
    double RescaleMap( vector<Vector3d>&pts );
    

protected:
    // params
    int _max_iterations;
    float _min_disparity;
    int _min_inliers;
    float _pose_init_thresh;

    // data
    Frame::Ptr _frame;
    vector<cv::Point2f> _px1, _px2;
    vector<cv::Point2f> _pt1, _pt2;
    Mat _H_estimated, _E_estimated;
    SE3 _T21;

    vector<bool> _inliers;
};

// useful tools 
Vector3d TriangulateFeatureNonLin( 
    const SE3& T, 
    const Vector3d& feature1, 
    const Vector3d& feature2 
); 

inline Vector2d project( const Vector3d& v ) {
    return Vector2d( v[0]/v[2], v[1]/v[2] );
}
    
inline Vector3d projectHomo( const Vector3d& v ) {
    return v/v[2];
}

}

#endif
