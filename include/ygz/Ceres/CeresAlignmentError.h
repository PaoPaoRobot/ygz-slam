#ifndef YGZ_CERES_ALIGNMENTERROR_H_
#define YGZ_CERES_ALIGNMENTERROR_H_

#include "ygz/Basic.h"
#include "ygz/Algorithm/CVUtils.h"

namespace ygz 
{
 
// 用于图像配准的Error，可以选择使用First Estimate Jacobian(同时也是逆向的，即在ref上计算)
class CeresAlignmentError: public ceres::SizedCostFunction<1, 2>
{
public:
    CeresAlignmentError( uint8_t* ref_patch, const Mat& curr_img, bool use_fej = true )
    : _ref_patch(ref_patch), _curr_img(curr_img), _use_fej(use_fej) {
        // 不管用不用都得算一下
        int step = 2*WarpHalfPatchSize;
        _fej[0] = _fej[1] = 0;
        for ( int i=0; i<PATTERN_SIZE; i++ ) {
            int ref_x = WarpPatchSize + PATTERN_DX[i];
            int ref_y = WarpPatchSize + PATTERN_DY[i];
            _fej[0] -= (_ref_patch[ ref_y*step+ref_x+1 ] - _ref_patch[ref_y*step+ref_x-1] )/255.0;
            _fej[1] -= (_ref_patch[ (ref_y+1)*step+ref_x ] - _ref_patch[(ref_y-1)*step+ref_x] )/255.0;
        }
    }
    
    void Enable( bool enable=true ) 
    {
        _enable = true; 
    }
    
    void SetFej( bool use_fej ) 
    { 
        _use_fej = use_fej; 
    }
    
    // 误差计算和雅可比计算
    // for k in pattern: Error(k) = I_ref(x) - I_cur(x+px)
    virtual bool Evaluate( double const* const* parameters, double* residuals, double** jacobians ) const override
    {
        bool setJacobian = jacobians&&jacobians[0];
        double curr_x = parameters[0][0];
        double curr_y = parameters[0][1];
        
        residuals[0] = 0;
        if ( setJacobian )
        {
            if ( _use_fej )
            {
                jacobians[0][0] = _fej[0];
                jacobians[0][1] = _fej[1];
            }
            else
                jacobians[0][0] = jacobians[0][1] = 0;
        }
        
        // compute the residuals 
        for ( int i=0; i<PATTERN_SIZE; i++ ) {
            
            double u = curr_x + PATTERN_DX[i];
            double v = curr_y + PATTERN_DY[i];
            
            int ref_x = WarpHalfPatchSize + PATTERN_DX[i];
            int ref_y = WarpHalfPatchSize + PATTERN_DY[i];
            
            if ( _enable && u>0 && v>0 && u<_curr_img.cols && v<_curr_img.rows ) {
                // 在图像中
                uchar gray = cvutils::GetBilateralInterpUchar(u,v,_curr_img);
                residuals[0] +=  (gray - _ref_patch[ ref_y*WarpPatchSize + ref_x ])/255.0;
                if ( setJacobian && _use_fej==false ) {
                    // 不用FEJ的话，就在current上面计算jacobian
                    double du = (cvutils::GetBilateralInterp(u+1, v, _curr_img) - cvutils::GetBilateralInterp(u-1,v,_curr_img))/2.0;
                    double dv = (cvutils::GetBilateralInterp(u, v+1, _curr_img) - cvutils::GetBilateralInterp(u,v-1,_curr_img))/2.0;
                    jacobians[0][0] += du;
                    jacobians[0][1] += dv;
                }
            } else {
                // residuals[i] = 0;
            }
        } 
        
        return true;
    }
    
private:
    uint8_t* _ref_patch=nullptr; 
    const Mat& _curr_img; 
    double _fej[2];
    bool _use_fej = true;
    bool _enable = true; 
};
    
}

#endif