#include "ygz/initializer.h"
#include "ygz/config.h"

#include <opencv2/calib3d/calib3d.hpp>

namespace ygz {
    
Initializer::Initializer()
: _max_iterations(10)
{
    _min_disparity = Config::get<float>("init.min_disparity");
    _min_inliers = Config::get<int>("init.min_inliers");
}

bool Initializer::TryInitialize(
    const vector< Vector2d >& px1, 
    const vector< Vector2d >& px2, 
    const Frame::Ptr& frame, 
    SE3& T12, 
    vector< Vector3d >& pts_triangulated )
{
    _frame = frame;
    // try essential and homography 
    for ( const Vector2d&v: px1 ) {
        _pt1.push_back( cv::Point2f(v[0], v[1]) );
    }
    for ( const Vector2d&v: px2 ) {
        _pt2.push_back( cv::Point2f(v[0], v[1]) );
    }
    
    bool retH = TestHomography();
    if ( retH == true ) {
        // when H is success, we use H, otherwise use F 
        // decompose the H 
        vector<HomographyDecomposition> decompose = DecomposeHomography();
        
        return true;
    }
    
    bool retE = TestEssential();
    return true;
}

bool Initializer::TestHomography()
{
    Mat inliers;
    Mat H = cv::findHomography( _pt1, _pt2, cv::RANSAC, 1, inliers );
    // count the inliers 
    int cnt_inliers(0); 
    _inliers.clear();
    for ( int i=0; i<inliers.rows; i++ ) {
        if ( inliers.at<int>(i) ) {
            cnt_inliers++;
            _inliers.push_back(true);
        } else {
            _inliers.push_back(false);
        }
    }
    
    LOG(INFO) << "number of inliers: " << cnt_inliers << endl;
    if ( cnt_inliers < _min_inliers ) {
        LOG(INFO) << "Too small inliers when finding H" << endl;
        return false; 
    }
    
    LOG(INFO) << "H estimated: " << H <<endl;
    _H_estimated = H;
    
    return true;
}

bool Initializer::TestEssential()
{
    Mat inliers;
    Mat E = cv::findEssentialMat( _pt1, _pt2, _frame->_camera->GetCameraMatrixCV(), 
                                  cv::RANSAC, 0.999, 1.0, inliers);
    // count the inliers 
    int cnt_inliers(0); 
    _inliers.clear();
    
    for ( int i=0; i<inliers.rows; i++ ) {
        if ( inliers.at<int>(i) ) {
            cnt_inliers++;
            _inliers.push_back(true);
        } else {
            _inliers.push_back(false);
        }
    }
    
    LOG(INFO) << "number of inliers: " << cnt_inliers << endl;
    if ( cnt_inliers < _min_inliers ) {
        LOG(INFO) << "Too small inliers when finding H" << endl;
        return false; 
    }
    
    LOG(INFO) << "E estimated: " << E <<endl;
    _E_estimated  = E;
    return true;
}

vector< HomographyDecomposition >&& Initializer::DecomposeHomography()
{
    // OpenCV's homography decomposition seems still not working, we directly use SVD
    // from SVO, I think case 2 and case 3 will not happen because we are using double matricies 
    vector<HomographyDecomposition> decompositions; 
    Eigen::Matrix3d H;
    H << _H_estimated.at<double>(0,0),  _H_estimated.at<double>(0,1), _H_estimated.at<double>(0,2),
         _H_estimated.at<double>(1,0),  _H_estimated.at<double>(1,1), _H_estimated.at<double>(1,2),
         _H_estimated.at<double>(2,0),  _H_estimated.at<double>(2,1), _H_estimated.at<double>(2,2);
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeThinU|Eigen::ComputeThinV); 
    Eigen::Vector3d sigma = svd.singularValues();
    double d1 = fabs( sigma[0] );
    double d2 = fabs( sigma[1] );
    double d3 = fabs( sigma[2] );
    
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    
    double s = U.determinant() * V.determinant();
    double dPrime_PM = d2;
    
    int nCase; 
    if ( d1 != d2 && d2!=d3 )
        nCase = 1; 
    else if ( d1==d2 && d2==d3 )
        nCase = 3; 
    else 
        nCase = 2;
    
    if(nCase != 1)
    {
        LOG(WARNING) << "FATAL Homography Initialization: This motion case is not implemented or is degenerate. Try again. " << endl;
        return std::move( decompositions );
    }
    
    double x1_PM;
    double x2;
    double x3_PM;
    
    // All below deals with the case = 1 case.
    // Case 1 implies (d1 != d3)
    {   // Eq. 12
        x1_PM = sqrt((d1*d1 - d2*d2) / (d1*d1 - d3*d3));
        x2    = 0;
        x3_PM = sqrt((d2*d2 - d3*d3) / (d1*d1 - d3*d3));
    };

    double e1[4] = {1.0,-1.0, 1.0,-1.0};
    double e3[4] = {1.0, 1.0,-1.0,-1.0};

    Vector3d np;
    HomographyDecomposition decomp;
    
    // Case 1, d' > 0:
    decomp.d = s * dPrime_PM;
    for(size_t signs=0; signs<4; signs++)
    {
        // Eq 13
        decomp.R = Eigen::Matrix3d::Identity();
        double dSinTheta = (d1 - d3) * x1_PM * x3_PM * e1[signs] * e3[signs] / d2;
        double dCosTheta = (d1 * x3_PM * x3_PM + d3 * x1_PM * x1_PM) / d2;
        decomp.R(0,0) = dCosTheta;
        decomp.R(0,2) = -dSinTheta;
        decomp.R(2,0) = dSinTheta;
        decomp.R(2,2) = dCosTheta;

        // Eq 14
        decomp.t[0] = (d1 - d3) * x1_PM * e1[signs];
        decomp.t[1] = 0.0;
        decomp.t[2] = (d1 - d3) * -x3_PM * e3[signs];

        np[0] = x1_PM * e1[signs];
        np[1] = x2;
        np[2] = x3_PM * e3[signs];
        decomp.n = V * np;

        decompositions.push_back(decomp);
    }

    // Case 1, d' < 0:
    decomp.d = s * -dPrime_PM;
    for(size_t signs=0; signs<4; signs++)
    {
        // Eq 15
        decomp.R = -1 * Eigen::Matrix3d::Identity();
        double dSinPhi = (d1 + d3) * x1_PM * x3_PM * e1[signs] * e3[signs] / d2;
        double dCosPhi = (d3 * x1_PM * x1_PM - d1 * x3_PM * x3_PM) / d2;
        decomp.R(0,0) = dCosPhi;
        decomp.R(0,2) = dSinPhi;
        decomp.R(2,0) = dSinPhi;
        decomp.R(2,2) = -dCosPhi;

        // Eq 16
        decomp.t[0] = (d1 + d3) * x1_PM * e1[signs];
        decomp.t[1] = 0.0;
        decomp.t[2] = (d1 + d3) * x3_PM * e3[signs];

        np[0] = x1_PM * e1[signs];
        np[1] = x2;
        np[2] = x3_PM * e3[signs];
        decomp.n = V * np;

        decompositions.push_back(decomp);
    }

    // Save rotation and translation of the decomposition
    for(unsigned int i=0; i<decompositions.size(); i++)
    {
        Eigen::Matrix3d R = s * U * decompositions[i].R * V.transpose();
        Vector3d t = U * decompositions[i].t;
        decompositions[i].T = Sophus::SE3(R, t);
    }
    return std::move( decompositions );
}


    
}
