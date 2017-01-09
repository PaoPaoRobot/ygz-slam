#ifndef YGZ_CERES_TYPES_H
#define YGZ_CERES_TYPES_H

#include <ceres/rotation.h>

#include "ygz/common_include.h"
#include "ygz/camera.h"
#include "ygz/utils.h"

using namespace ygz::utils;

namespace ygz
{

// 既有位姿又有点的重投影
class CeresReprojectionError
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW 
    CeresReprojectionError( const Vector2d& pt_cam ): _pt_cam(pt_cam) { }
    
    // cost = z - (Rp+t), note the first three components in the pose are translation 
    template< typename T> 
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
        
        if ( p[2] < T(0) ) {
            residuals[0] = residuals[1] = T(0);
            return false;
        }
        
        residuals[0] = _pt_cam[0] - p[0]/p[2];
        residuals[1] = _pt_cam[1] - p[1]/p[2];
        return true; 
    }
protected:
    Vector2d _pt_cam; // observation: normalized camera coordinate 
};

// 只有位姿的重投影
// 加一个使能，可以定义这条误差是否参与计算
class CeresReprojectionErrorPoseOnly
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW 
    CeresReprojectionErrorPoseOnly( 
        const Vector2d& pt_cam, const Vector3d& pt_world, bool enable = true
    ): _pt_cam(pt_cam), _pt_world(pt_world), _enable(enable) { }
    
    void SetEnable( bool enable = true ) {
        _enable = enable;
    }
    
    // cost = z - (Rp+t), note the first three components in the pose are translation 
    template< typename T > 
    bool operator() ( 
        const T* const pose_TCW, 
        T* residuals
    ) const {
        if ( _enable == false ) {
            residuals[0] = residuals[1] = T(0);
            return true;
        }
        
        T p[3];
        T pw[3];
        T rot[3];
        for ( size_t i=0; i<3; i++ )
            rot[i] = pose_TCW[i+3];
        for ( size_t i=0; i<3; i++ )
            pw[i] = T(_pt_world[i]);
        ceres::AngleAxisRotatePoint<T>( rot, pw, p );
        
        p[0] += pose_TCW[0]; 
        p[1] += pose_TCW[1]; 
        p[2] += pose_TCW[2]; 
        
        if ( p[2] < T(0) ) {
            residuals[0] = residuals[1] = T(0);
            return false;
        }
        
        residuals[0] = _pt_cam[0] - p[0]/p[2];
        residuals[1] = _pt_cam[1] - p[1]/p[2];
        return true; 
    }
protected:
    Vector2d _pt_cam; // observation: normalized camera coordinate 
    Vector3d _pt_world; // 3D point in world frame 
    bool _enable; 
};

// 反之，只有点的重投影
// Error = pt - TCR*pw
class CeresReprojectionErrorPointOnly
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW 
    CeresReprojectionErrorPointOnly( 
        const Vector2d& pt_cam, const SE3& TCW
    ): _pt_cam(pt_cam)  {
        Vector3d r = TCW.so3().log(), t=TCW.translation();
        // 前面三项为t，后面三项为r 
        _TCW.head<3>() = t;
        _TCW.tail<3>() = r;
    }
    
    // cost = z - (Rp+t), note the first three components in the pose are translation 
    template< typename T > 
    bool operator() ( 
        const T* const point_world, 
        T* residuals
    ) const {
        T p[3];
        T rot[3];
        for ( size_t i=0; i<3; i++ )
            rot[i] = (T) _TCW[i+3];
        ceres::AngleAxisRotatePoint<T>( rot, point_world, p );
        
        p[0] += (T) _TCW[0]; 
        p[1] += (T) _TCW[1]; 
        p[2] += (T) _TCW[2]; 
        
        if ( p[2] < T(0) ) {
            residuals[0] = residuals[1] = T(0);
            return false;
        }
        
        residuals[0] = _pt_cam[0] - p[0]/p[2];
        residuals[1] = _pt_cam[1] - p[1]/p[2];
        return true; 
    }
protected:
    Vector2d _pt_cam; // observation: normalized camera coordinate 
    Vector6d _TCW; // 3D point in world frame 
};

// used in sparse direct method 
// the pattern is inspired by DSO
// 第一个量是6自由度pose，第二个量是深度值
class CeresReprojSparseDirectError: public ceres::SizedCostFunction<PATTERN_SIZE,6,1>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    CeresReprojSparseDirectError ( 
        cv::Mat& curr_img, 
        PixelPattern& ref_pattern,
        Vector3d& pt_ref, 
        PinholeCamera* cam,
        double scale
    ) : _curr_img(curr_img), _pt_ref(pt_ref), _cam(cam), _ref_pattern(ref_pattern),_scale(scale)
    { 
        
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
            
            // 前三维是平移，后三维是李代数 so(3)，这样可以少算一个 J
            SE3 TCR = SE3 (
                SO3::exp( pose.tail<3>() ), 
                pose.head<3>()
            );
            
            double depth = parameters[1][0];  // 第二个参数为d
            
            // LOG(INFO) << "TCR = "<<endl<<TCR.matrix()<<endl;
            Vector3d pt_curr = TCR*(_pt_ref*depth);
            Vector2d px_curr = _cam->Camera2Pixel( pt_curr ) / _scale; // consider the pyramid 
            
            bool setJacobian = false;
            if ( jacobians && jacobians[0] ) 
                setJacobian = true; 
            bool visible = IsInside( px_curr, _curr_img ) && pt_curr[2]>0;
            
            for ( int i=0; i<PATTERN_SIZE; i++ ) {
                
                double u = px_curr[0] + PATTERN_DX[i];
                double v = px_curr[1] + PATTERN_DY[i];
                
                if ( visible ) {
                    residuals[i] = _ref_pattern.pattern[i] - utils::GetBilateralInterp(u,v,_curr_img);
                    if ( setJacobian ) {
                        double du = (GetBilateralInterp(u+1, v, _curr_img) - GetBilateralInterp(u-1,v,_curr_img))/2;
                        double dv = (GetBilateralInterp(u, v+1, _curr_img) - GetBilateralInterp(u,v-1,_curr_img))/2;
                        Eigen::Vector2d duv(du,dv);
                        Eigen::Matrix<double,2,6> J_uv_xyz = utils::JacobXYZ2Pixel(pt_curr, _cam);
                        Vector6d J = duv.transpose()*J_uv_xyz;
                        // 这个雅可比形状很难说清，参见 http://ceres-solver.org/nnls_modeling.html#costfunction
                        // 总之我是调了一阵才发现这个形状是对的
                        for ( int k=0; k<6; k++ )
                        {
                            jacobians[0][i*6+k] = J[k]*_scale;
                        }
                        // jacobians[1][i] = duv.transpose()*J_uv_xyz.block<2,3>(0,0)*( TCR*_pt_ref );
                        jacobians[1][i] = 0;
                    }
                } else { // 点在图像外面，不考虑它带来的误差
                    residuals[i] = 0;
                    if ( setJacobian ) {
                        for ( int k=0; k<6; k++ )
                        {
                            jacobians[0][i*6+k] = 0;
                        }
                        jacobians[1][i] = 0;
                    }
                }
            }
            
            return true;
        }
protected:
    cv::Mat& _curr_img; 
    PixelPattern& _ref_pattern;
    Vector3d _pt_ref;
    PinholeCamera* _cam =nullptr;
    double _scale;
};

// 用于 alignment 的 Error 项
// 也照着 DSO 那样用 pattern，不过估计的是像素位置，所以 parameter 是2
// 可以使用First estimate jacobian啦，FEJ会稍微快一点儿
class CeresAlignmentError: public ceres::SizedCostFunction<PATTERN_SIZE, 2> 
{
public:
    static const uint HALF_PATCH_REF=4; 
    CeresAlignmentError( uint8_t* ref_patch, const Mat& curr_img, bool use_fej = true )
        : _ref_patch(ref_patch), _curr_img(curr_img), _use_fej(use_fej) {
            if ( use_fej ) {
                int step = 2*HALF_PATCH_REF;
                for ( int i=0; i<PATTERN_SIZE; i++ ) {
                    int ref_x = HALF_PATCH_REF + PATTERN_DX[i];
                    int ref_y = HALF_PATCH_REF + PATTERN_DY[i];
                    _fej[i][0] = (_ref_patch[ ref_y*step+ref_x+1 ] - _ref_patch[ref_y*step+ref_x-1] );
                    _fej[i][1] = (_ref_patch[ (ref_y+1)*step+ref_x ] - _ref_patch[(ref_y-1)*step+ref_x] );
                }
            }
        }
    void SetFej( bool use_fej ) { _use_fej = use_fej; }
        
    virtual bool Evaluate( 
        double const* const* parameters,
        double* residuals,
        double** jacobians) const override
        {
            bool setJacobian = false; 
            if ( jacobians && jacobians[0] ) 
                setJacobian = true; 
            
            double curr_x = parameters[0][0];
            double curr_y = parameters[0][1];
            
            for ( int i=0; i<PATTERN_SIZE; i++ ) {
                double u = curr_x + PATTERN_DX[i];
                double v = curr_y + PATTERN_DY[i];
                int ref_x = HALF_PATCH_REF + PATTERN_DX[i];
                int ref_y = HALF_PATCH_REF + PATTERN_DY[i];
                
                if ( u>0 && v>0 && u<_curr_img.cols && v<_curr_img.rows ) {
                    // 在图像中
                    uchar gray = GetBilateralInterpUchar(u,v,_curr_img);
                    residuals[i] =  gray - _ref_patch[ ref_y*2*HALF_PATCH_REF + ref_x ];
                    if ( setJacobian ) {
                        if ( _use_fej ) {
                            jacobians[0][i*2]   =   _fej[i][0];
                            jacobians[0][i*2+1] =   _fej[i][1];
                        } else {
                            // 不用FEJ的话，就在current上面计算jacobian
                            double du = (GetBilateralInterpUchar(u+1, v, _curr_img) - GetBilateralInterpUchar(u-1,v,_curr_img))/2;
                            double dv = (GetBilateralInterpUchar(u, v+1, _curr_img) - GetBilateralInterpUchar(u,v-1,_curr_img))/2;
                            jacobians[0][i*2] = du;
                            jacobians[0][i*2+1] = dv;
                        }
                    }
                } else {
                    residuals[i] = 0;
                    if ( setJacobian ) {
                        jacobians[0][i*2] = jacobians[0][i*2+1] = 0;
                    }
                }
            }
            return true;
        }
    
private:
    uint8_t* _ref_patch;
    const Mat& _curr_img;
    double _fej[PATTERN_SIZE][2];  // first estimate jacobian 
    bool _use_fej;
};

}
#endif
