#ifndef YGZ_CERES_TYPES_H
#define YGZ_CERES_TYPES_H

#include <ceres/rotation.h>

#include "ygz/common_include.h"
#include "ygz/camera.h"
#include "ygz/utils.h"

using namespace ygz::utils;

namespace ygz
{

class CeresReprojectionError
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW 
    CeresReprojectionError( const Vector2d& pt_cam ): _pt_cam(pt_cam) { }
    
    // cost = z - (Rp+t), note the first three components in the pose are translation 
    template< typename T=double > 
    bool operator() ( 
        const T* const pose_TCW, 
        const T* const point_world, 
        T* residuals
    ) const {
        T p[3];
        T rot[3];
        for ( size_t i=0; i<3; i++ )
            rot[i] = pose_TCW[i+3];
        ceres::AngleAxisRotatePoint<T>( rot, point_world, p );
        
        p[0] += pose_TCW[0]; 
        p[1] += pose_TCW[1]; 
        p[2] += pose_TCW[2]; 
        
        residuals[0] = _pt_cam[0] - p[0]/p[2];
        residuals[1] = _pt_cam[1] - p[1]/p[2];
        return true; 
    }
protected:
    Vector2d _pt_cam; // observation: normalized camera coordinate 
};

// used in sparse direct method 
// the pattern is inspired by DSO
class CeresReprojSparseDirectError: public ceres::SizedCostFunction<PATTERN_SIZE,6>
{
public:
    CeresReprojSparseDirectError ( 
        cv::Mat& curr_img, 
        PixelPattern& ref_pattern,
        Vector3d& pt_ref, 
        PinholeCamera::Ptr cam
    ) : _curr_img(curr_img), _pt_ref(pt_ref), _cam(cam), _ref_pattern(ref_pattern) {
    }
    
    // evaluate the residual and jacobian 
    // Eq: I_C ( T_CR*P_R ) - I_R ( P_R )
    virtual bool Evaluate( 
        double const* const* parameters,
        double* residuals,
        double** jacobians) const override
        {
            Vector6d pose; 
            for ( size_t i=0; i<6; i++ )
                pose[i] = parameters[0][i];
            SE3 TCR = SE3::exp(pose);
            // LOG(INFO) << "TCR = "<<endl<<TCR.matrix()<<endl;
            Vector3d pt_curr = TCR*_pt_ref;
            Vector2d px_curr = _cam->Camera2Pixel( pt_curr );
            
            bool setJacobian = false;
            if ( jacobians && jacobians[0] ) 
                setJacobian = true; 
            
            for ( int i=0; i<PATTERN_SIZE; i++ ) {
                double u = px_curr[0] + PATTERN_DX[i];
                double v = px_curr[1] + PATTERN_DY[i];
                if ( u>0 && v>0 && u<_curr_img.cols && v<_curr_img.rows )
                {
                    residuals[i] = _ref_pattern.pattern[i] - utils::GetBilateralInterp(u,v,_curr_img);
                    
                    // LOG(INFO) << "ref pixel = "<<_ref_pattern.pattern[i] << ", curr pixel = "<< utils::GetBilateralInterp(u,v,_curr_img) << endl;
                    
                    if ( setJacobian ) {
                        double du = (GetBilateralInterp(u+1, v, _curr_img) - GetBilateralInterp(u-1,v,_curr_img))/2;
                        double dv = (GetBilateralInterp(u, v+1, _curr_img) - GetBilateralInterp(u,v-1,_curr_img))/2;
                        
                        Eigen::Vector2d duv(du,dv);
                        Eigen::Matrix<double,2,6> J_uv_xyz = utils::JacobXYZ2Pixel(pt_curr, _cam);
                        Vector6d J = duv.transpose()*J_uv_xyz;
                        // LOG(INFO) << "J=" << J.transpose() << endl;
                      
                        for ( int k=0; k<6; k++ )
                            jacobians[0][k*PATTERN_SIZE+i] = J[k];
                    }
                }
            }
            return true;
        }
    
    /*
    template<typename T>
    bool operator() (
        const T* const pose_TCR,
        T* residuals 
    ) const {
        T p[3];
        T rot[3];
        for ( size_t i=0; i<3; i++ )
            rot[i] = pose_TCR[i+3];
        T pt_ref[3];
        for ( size_t i=0; i<3; i++ )
            pt_ref[i] = _pt_ref[i];
        ceres::AngleAxisRotatePoint<T>( rot, pt_ref, p );
        p[0] += pose_TCR[0]; 
        p[1] += pose_TCR[1]; 
        p[2] += pose_TCR[2]; 
        
        T px[2];
        _cam->Camera2Pixel<T>(p,px);
        
        T res = 0; 
        for ( int i=0; i<PATTERN_SIZE; i++ ) {
            T u = px[0] + PATTERN_DX[i];
            T v = px[1] + PATTERN_DY[i];
            res += _ref_pattern[i] - utils::getBilateralInterp(u,v,_curr_img);
        }
        residuals[0] = res;
        return true; 
    }
    */
    
protected:
    cv::Mat& _curr_img; 
    PixelPattern& _ref_pattern;
    Vector3d _pt_ref;
    PinholeCamera::Ptr _cam =nullptr;
};

}
#endif
