#ifndef YGZ_CERES_REPROJECTION_SPARSE_DIRECT_ERROR_H_
#define YGZ_CERES_REPROJECTION_SPARSE_DIRECT_ERROR_H_

#include "ygz/Basic.h"
#include "ygz/Algorithm/CVUtils.h"


namespace ygz 
{
    
// used in model based sparse alignment
// the pattern is inspired by DSO
// 第一个量是6自由度pose，第二个量是深度值
// 可以选择是否使用First estimate jacobian
// 这个在FeJ下仍然有些慢，看看能否有加速的手段
// 可能像LSD或DSO那样考虑单个像素会更简单一些
class CeresReprojSparseDirectError: public ceres::SizedCostFunction<PATTERN_SIZE,6>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    CeresReprojSparseDirectError 
    ( 
        const cv::Mat& ref_img,
        const cv::Mat& curr_img, 
        uchar* ref_pattern,
        const Vector2d& px_ref,
        const Vector3d& pt_ref, 
        PinholeCamera* cam,
        const double& scale,
        bool use_fej = true
    ) : _curr_img(curr_img), _pt_ref(pt_ref), 
        _cam(cam), _ref_pattern(ref_pattern), _scale(scale), _use_fej(use_fej)
    { 
        // 计算FeJ
        Vector2d px_ref_scaled = px_ref/_scale;
        for ( int i=0; i<PATTERN_SIZE; i++ )
        {
            double u = px_ref_scaled[0] + PATTERN_DX[i];
            double v = px_ref_scaled[1] + PATTERN_DY[i];
            _fej[i][0] = (cvutils::GetBilateralInterpUchar(u+1, v, ref_img) - cvutils::GetBilateralInterpUchar(u-1,v,ref_img))/2;
            _fej[i][1] = (cvutils::GetBilateralInterpUchar(u, v+1, ref_img) - cvutils::GetBilateralInterpUchar(u,v-1,ref_img))/2;
        }
    }
        
    // evaluate the residual and jacobian 
    // Eq: I_C ( T_CR*P_R ) - I_R ( P_R )
    virtual bool Evaluate( 
        double const* const* parameters,
        double* residuals,
        double** jacobians
    ) const override
    {
        Vector6d pose; 
        for ( size_t i=0; i<6; i++ )
            pose[i] = parameters[0][i];
        
        // 前三维是平移，后三维是李代数 so(3)，这样可以少算一个 J
        SE3 TCR = SE3 (
            SO3::exp( pose.tail<3>() ), 
            pose.head<3>()
        );
        
        // LOG(INFO) << "TCR = "<<endl<<TCR.matrix()<<endl;
        Vector3d pt_curr = TCR*_pt_ref;
        Vector2d px_curr = _cam->Camera2Pixel( pt_curr ) / _scale; // consider the pyramid 
        
        bool setJacobian = false;
        if ( jacobians && jacobians[0] ) 
            setJacobian = true; 
        
        bool visible = px_curr[0]>=5 && px_curr[1]>=5 && 
            px_curr[0]<_curr_img.cols-5 && px_curr[1] < _curr_img.rows-5
            &&  pt_curr[2]>0;
        
        for ( int i=0; i<PATTERN_SIZE; i++ ) {
            
            double u = px_curr[0] + PATTERN_DX[i];
            double v = px_curr[1] + PATTERN_DY[i];
            
            if ( visible ) 
            {
                residuals[i] = _ref_pattern[i] - cvutils::GetBilateralInterpUchar(u,v,_curr_img);
                if ( setJacobian ) 
                {
                    Vector2d duv;
                    if ( _use_fej == false )
                    {
                        duv[0] = (cvutils::GetBilateralInterpUchar(u+1, v, _curr_img) - cvutils::GetBilateralInterpUchar(u-1,v,_curr_img))/2;
                        duv[1] = (cvutils::GetBilateralInterpUchar(u, v+1, _curr_img) - cvutils::GetBilateralInterpUchar(u,v-1,_curr_img))/2;
                    }
                    else 
                    {
                        duv = _fej[i];
                    }
                    
                    Eigen::Matrix<double,2,6> J_uv_xyz = cvutils::JacobXYZ2Pixel(pt_curr, _cam);
                    Vector6d J = duv.transpose()*J_uv_xyz;
                    
                    // 这个雅可比形状很难说清，参见 http://ceres-solver.org/nnls_modeling.html#costfunction
                    // 总之我是调了一阵才发现这个形状是对的
                    for ( int k=0; k<6; k++ )
                    {
                        jacobians[0][i*6+k] = J[k]*_scale;
                    }
                }
            } else { // 点在图像外面或深度为负，不考虑它带来的误差
                residuals[i] = 0;
                if ( setJacobian ) {
                    for ( int k=0; k<6; k++ )
                    {
                        jacobians[0][i*6+k] = 0;
                    }
                }
            }
        }
        return true;
    }
    
protected:
    const cv::Mat& _curr_img; 
    uchar* _ref_pattern;        // 这个需要按照pattern顺序来排列
    const Vector3d _pt_ref;
    PinholeCamera* _cam =nullptr;
    double _scale;
    
    Vector2d _fej[PATTERN_SIZE];        // First Estimate Jacobians (inverse)
    bool _use_fej;                      // set true to enable FeJ, will be faster
}; 
    
}

#endif