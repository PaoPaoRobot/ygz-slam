#include <opencv2/calib3d/calib3d.hpp>

#include "ygz/initializer.h"
#include "ygz/config.h"
#include "ygz/memory.h"
#include "ygz/utils.h"

namespace ygz {

Initializer::Initializer()
    : _max_iterations(10), _pose_init_thresh(2.0)
{
    _min_disparity = Config::get<float>("init.min_disparity");
    _min_inliers = Config::get<int>("init.min_inliers");
}

bool Initializer::TryInitialize(
    const vector< Vector2d >& px1,
    const vector< Vector2d >& px2,
    Frame::Ptr& ref,
    Frame::Ptr& curr
)
{
    _frame = curr;
    // try essential and homography
    // 计算 两个相机坐标系下的点
    for ( const Vector2d&v: px1 ) {
        _px1.push_back( cv::Point2f(v[0], v[1]) );
        Vector3d pt = ref->_camera->Pixel2Camera( v );
        _pt1.push_back( cv::Point2f(pt[0], pt[1]) );
    }
    for ( const Vector2d&v: px2 ) {
        _px2.push_back( cv::Point2f(v[0], v[1]));
        Vector3d pt = _frame->_camera->Pixel2Camera( v );
        _pt2.push_back( cv::Point2f(pt[0], pt[1]) );
    }

    // test E and H
    bool retE = TestEssential();
    bool retH = TestHomography();

    SE3 T21_E, T21_H;

    if ( retE == true ) {
        // recover SE3 from E
        // TODO: check if the order is right!
        Mat R, t;
        int ret = cv::recoverPose( _E_estimated, _px1, _px2, _frame->_camera->GetCameraMatrixCV(), R, t);
        Eigen::Matrix3d RR;
        RR << R.at<double>(0,0) , R.at<double>(0,1) , R.at<double>(0,2),
           R.at<double>(1,0) , R.at<double>(1,1) , R.at<double>(1,2),
           R.at<double>(2,0) , R.at<double>(2,1) , R.at<double>(2,2);
        T21_E = SE3(
                    RR,
                    Vector3d( t.at<double>(0,0), t.at<double>(1,0), t.at<double>(2,0) )
                );

        LOG(INFO) << "T21 from E = \n" << T21_E.matrix() << endl;
    }
    
    if (retH == true ) {
        // decompose the H
        vector<HomographyDecomposition> decompose = DecomposeHomography();
        if ( !decompose.empty() ) {
            // doesnot degenerate
            FindBestDecomposition( decompose );
            T21_H = decompose[0].T;
            LOG(INFO) << "T21 from H = \n" << T21_H.matrix() << endl;
        } else {
            retH = false;
        }
        // decompose H failed, although this does not often happen, turn to E for initialization
    }

    // initialization success, triangulate the tracked points
    // select from E and H
    // TODO compare the result of H and E, now we only select E
    vector<Vector3d> features3d_curr;
    if ( retE == false && retH == false )
        return false;
    if ( retE == true ) {
        Triangulate( T21_E, features3d_curr );
        // rescale the map 
        double scale = RescaleMap( features3d_curr );
        _frame->_T_c_w = T21_E * ref->_T_c_w;
        // change the scale between ref and current 
        _frame->_T_c_w.translation() = -_frame->_T_c_w.rotation_matrix() * ( ref->Pos() + 1/scale*( _frame->Pos() - ref->Pos()));
        
        // add feature into both ref and curr frame 
        SE3 T_w_c = _frame->_T_c_w.inverse(); 
        
        // register both the current 
        // Memory::RegisterFrame( ref );
        Memory::RegisterFrame( curr );
        
        // set the map points 
        for ( size_t i=0; i<_inliers.size(); i++ ) {
            if ( _inliers[i] == false ) continue; 
            if ( ref->InFrame(px1[i], 10) && curr->InFrame(px2[i]) && features3d_curr[i][2]>0 ) {
                Vector3d pos = T_w_c * ( features3d_curr[i]/scale );
                
                MapPoint::Ptr map_point = Memory::CreateMapPoint();
                map_point->_pos_world = pos;
                map_point->_bad = false;
                
                // add the observations into frame 
                ref->_map_point.push_back( map_point->_id );
                map_point->_obs[ ref->_id ] = Vector3d( _px1[i].x, _px1[i].y, 1 );
                
                curr->_map_point.push_back( map_point->_id );
                map_point->_obs[ curr->_id ] = Vector3d( _px2[i].x, _px2[i].y, features3d_curr[i][2]/scale );
            }
        }
        return true; 
    } else if ( retH == true ) {
        Triangulate( T21_H, features3d_curr );
        // rescale the map 
        double scale = RescaleMap( features3d_curr );
        _frame->_T_c_w = T21_H * ref->_T_c_w;
        // change the scale between ref and current 
        _frame->_T_c_w.translation() = -_frame->_T_c_w.rotation_matrix() * ( ref->Pos() + 1/scale*( _frame->Pos() - ref->Pos()));
        
        // add feature into both ref and curr frame 
        SE3 T_w_c = _frame->_T_c_w.inverse(); 
        for ( size_t i=0; i<_inliers.size(); i++ ) {
            if ( _inliers[i] == false ) continue; 
            if ( ref->InFrame(px1[i], 10) && curr->InFrame(px2[i], 10) && features3d_curr[i][2]>0 ) {
                Vector3d pos = T_w_c * ( features3d_curr[i]*1.0/scale );
                
                MapPoint::Ptr map_point = Memory::CreateMapPoint();
                map_point->_pos_world = pos;
                map_point->_bad = false;
                
                // add the observations into frame 
                ref->_map_point.push_back( map_point->_id );
                map_point->_obs[ ref->_id ] = Vector3d( _px1[i].x, _px1[i].y, 1 );
                
                curr->_map_point.push_back( map_point->_id );
                map_point->_obs[ curr->_id ] = Vector3d( _px2[i].x, _px2[i].y, features3d_curr[i][2]/scale );
            }
        }
        return true; 
    }
    return false;
}

void Initializer::Triangulate(const SE3& T21, vector< Vector3d >& pts_triangulated)
{
    // triangulate the inliers
    int cnt_inlier = 0;
    for ( size_t i=0; i<_pt2.size(); i++ ) {
        // if ( _inliers[i] == false ) continue;
        Vector3d p1 = Vector3d( _pt1[i].x, _pt1[i].y, 1);
        Vector3d p2 = Vector3d( _pt2[i].x, _pt2[i].y, 1 );

        Vector3d p = TriangulateFeatureNonLin( T21, p2, p1 );
        pts_triangulated.push_back( p );
        // compute the reprojection error to remove the outlier
        double error1 = (p2 - projectHomo(p) ).norm();
        double error2 = (p1 - projectHomo(T21.inverse()*p) ).norm();
        if ( error1 > _pose_init_thresh || error2 > _pose_init_thresh ) {
            // set as outlier
            _inliers[i] = false;
        } else {
            _inliers[i] = true;
            cnt_inlier++;
        }
    }
    LOG(INFO) << "inliers in triangulation: " << cnt_inlier << endl;
}

double Initializer::RescaleMap(vector< Vector3d >& pts)
{
    double mean_d = 0;
    int cnt = 0;
    for ( size_t i=0; i<pts.size(); i++ ) {
        if ( _inliers[i]==true ) {
            mean_d += pts[i][2];
            cnt++;
        }
    }
    mean_d /= double(cnt);
    return mean_d;
}


bool Initializer::TestHomography()
{
    Mat inliers;
    // note the homography is computed by normalized coordinates
    Mat H = cv::findHomography( _pt1, _pt2, cv::RANSAC, 1, inliers );

    // test the H
    Mat res = (cv::Mat_<double>(3,1) << _pt2[0].x, _pt2[0].y, 1) - H *
              (cv::Mat_<double>(3,1) << _pt1[0].x, _pt1[0].y, 1 );
    LOG(INFO) << "res = " << res <<endl;

    Mat res2 = (cv::Mat_<double>(3,1) << _pt1[0].x, _pt1[0].y, 1) - H *
               (cv::Mat_<double>(3,1) << _pt2[0].x, _pt2[0].y, 1 );
    LOG(INFO) << "res2 = " << res2 <<endl;

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

    // LOG(INFO) << "H type = " << H.type() << ", CV_64F = " << CV_64F << ", 32F = " << CV_32F << endl;

    return true;
}

bool Initializer::TestEssential()
{
    Mat inliers;
    Mat E = cv::findEssentialMat( _px1, _px2, _frame->_camera->GetCameraMatrixCV(),
                                  cv::RANSAC, 0.999, 1.0, inliers);
    // Mat E = cv::findEssentialMat( _pt1, _pt2, _frame->_camera->GetCameraMatrixCV());
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

vector< HomographyDecomposition > Initializer::DecomposeHomography()
{
    // OpenCV's homography decomposition seems still not work, we directly use SVD
    // from SVO, I think case 2 and case 3 will not happen because we are using double matricies
    vector<HomographyDecomposition> decompositions;
    Eigen::Matrix3d H;
    H << _H_estimated.at<double>(0,0),  _H_estimated.at<double>(0,1), _H_estimated.at<double>(0,2),
      _H_estimated.at<double>(1,0),  _H_estimated.at<double>(1,1), _H_estimated.at<double>(1,2),
      _H_estimated.at<double>(2,0),  _H_estimated.at<double>(2,1), _H_estimated.at<double>(2,2);
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(H, Eigen::ComputeThinU|Eigen::ComputeThinV);
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
        LOG(INFO) << "decomposing H returns "<<endl<<decompositions[i].T.matrix()<<endl;
    }
    return decompositions;
}

void Initializer::FindBestDecomposition( vector< HomographyDecomposition >& decomp )
{
    assert( decomp.size() == 8 );
    for ( size_t i=0; i<decomp.size(); i++ ) {
        HomographyDecomposition& decom = decomp[i];
        size_t nPositive = 0;
        for ( size_t m=0; m<_pt1.size(); m++ ) {
            if ( _inliers[m] == false )
                continue;
            Vector2d v2( _pt1[m].x, _pt1[m].y );
            double visible = ( _H_estimated.at<double>(2,0)*v2[0] + _H_estimated.at<double>(2,1) * v2[1] + _H_estimated.at<double>(2,2) ) / decom.d;
            if ( visible > 0 )
                nPositive++;
        }
        decom.score = -nPositive;
    }

    // 也真是醉了，对一个vector在做erase，还好只有八个
    sort( decomp.begin(), decomp.end(),
    [&](const HomographyDecomposition& h1, const HomographyDecomposition& h2) {
        return h1.score < h2.score;
    });

    decomp.resize(4);

    for ( HomographyDecomposition& d: decomp ) {
        LOG(INFO) << "selected T: \n"<<d.T.matrix()<<endl;
    }

    for ( size_t i=0; i<decomp.size(); i++ ) {
        HomographyDecomposition& decom = decomp[i];
        size_t nPositive = 0;
        for ( size_t m=0; m<_pt1.size(); m++ ) {
            if ( _inliers[m] == false )
                continue;
            Vector3d v3( _pt1[m].x, _pt1[m].y, 1 );
            double visible = v3.dot( decom.n ) / decom.d;
            if ( visible > 0 )
                nPositive++;
        }
        decom.score = -nPositive;
    }

    sort( decomp.begin(), decomp.end(),
    [&](const HomographyDecomposition& h1, const HomographyDecomposition& h2) {
        return h1.score < h2.score;
    });
    decomp.resize(2);

    for ( HomographyDecomposition& d: decomp ) {
        LOG(INFO) << "selected T: \n"<<d.T.matrix()<<endl;
        LOG(INFO) << "score: "<<d.score<<endl;

    }

    double dRatio = (double) decomp[1].score / (double) decomp[0].score;

    if(dRatio < 0.9) // no ambiguity!
        decomp.erase(decomp.begin() + 1);
    else {
        // two-way ambiguity. Resolve by sampsonus score of all points.
        double dErrorSquaredLimit  = _pose_init_thresh * _pose_init_thresh * 4;
        double adSampsonusScores[2];
        for(size_t i=0; i<2; i++) {
            Sophus::SE3 T = decomp[i].T;
            // Eigen::Matrix3d Essential = T.rotation_matrix() * SO3::hat(T.translation());
            Eigen::Matrix3d Essential = SO3::hat(T.translation()) * T.rotation_matrix();
            double dSumError = 0;
            for(size_t m=0; m < _pt1.size(); m++ ) {
                // if ( _inliers[m] == false )
                // continue;
                double d = SampSonusError( _pt2[m], Essential, _pt1[m]);
                if(d > dErrorSquaredLimit)
                    d = dErrorSquaredLimit;
                dSumError += d;
            }
            adSampsonusScores[i] = dSumError;
        }

        LOG(INFO) << "score: "<<adSampsonusScores[0]<<", "<<adSampsonusScores[1] << endl;

        if(adSampsonusScores[0] <= adSampsonusScores[1])
            decomp.erase(decomp.begin() + 1);
        else
            decomp.erase(decomp.begin());
    }

}

double Initializer::SampSonusError(
    const cv::Point2f& v2Dash, const Eigen::Matrix3d& Essential, const cv::Point2f& v2)
{
    Vector3d v3Dash (v2Dash.x, v2Dash.y, 1 );
    Vector3d v3 (v2.x, v2.y, 1);

    double dError = v3Dash.transpose() * Essential * v3;

    Vector3d fv3 = Essential * v3;
    Vector3d fTv3Dash = Essential.transpose() * v3Dash;

    Vector2d fv3Slice = fv3.head<2>();
    Vector2d fTv3DashSlice = fTv3Dash.head<2>();
    return (dError * dError / (fv3Slice.dot(fv3Slice) + fTv3DashSlice.dot(fTv3DashSlice)));
}

Vector3d TriangulateFeatureNonLin(const SE3& T, const Vector3d& feature1, const Vector3d& feature2)
{
    Vector3d f2 = T.rotation_matrix() * feature2;
    Vector2d b;
    b[0] = T.translation().dot(feature1);
    b[1] = T.translation().dot(f2);
    Eigen::Matrix2d A;
    A(0,0) = feature1.dot(feature1);
    A(1,0) = feature1.dot(f2);
    A(0,1) = -A(1,0);
    A(1,1) = -f2.dot(f2);
    Vector2d lambda = A.inverse() * b;
    Vector3d xm = lambda[0] * feature1;
    Vector3d xn = T.translation() + lambda[1] * f2;
    return ( xm + xn )/2;
}


}
