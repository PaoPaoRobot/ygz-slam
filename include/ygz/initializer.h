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
    // NOTE px1 and px2 are normalized coordinates, not pixel coordinates!
    bool TryInitialize(
        const vector<Vector2d>& px1,
        const vector<Vector2d>& px2,
        const Frame::Ptr& frame,
        SE3& T12,
        vector<Vector3d>& pts_triangulated
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
    vector<HomographyDecomposition>&& DecomposeHomography();


protected:
    // params
    int _max_iterations;
    float _min_disparity;
    int _min_inliers;

    // data
    Frame::Ptr _frame;
    vector<cv::Point2f> _pt1, _pt2;
    Mat _H_estimated, _E_estimated;

    vector<bool> _inliers;
};

}

#endif
