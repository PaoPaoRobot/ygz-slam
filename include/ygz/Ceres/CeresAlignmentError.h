#ifndef YGZ_CERES_ALIGNMENTERROR_H_
#define YGZ_CERES_ALIGNMENTERROR_H_

#include "ygz/Basic.h"
#include "ygz/Algorithm/CVUtils.h"

namespace ygz 
{
 
// 用于图像配准的Error，可以选择使用First Estimate Jacobian(同时也是逆向的，即在ref上计算)
class CeresAlignmentError: public ceres::SizedCostFunction<WarpPatchSize*WarpPatchSize, 2> // 默认为64的patch
{
public:
    CeresAlignmentError ( 
        uint8_t* ref_patch, 
        uint8_t* ref_patch_with_boarder,        // 有boarder的这个用来算梯度，它是10x10的，否则一开始那个fej没法算
        const Mat& curr_img, 
        bool use_fej = true 
    )
    : _ref_patch(ref_patch), _curr_img(curr_img), _use_fej(use_fej) {
        // 不管用不用都得算一下fej 
        int step = WarpPatchSize+2;
        for ( int x=0; x<WarpPatchSize; x++ )
            for ( int y=0; y<WarpPatchSize; y++ ) 
            {
                _fej[y*WarpPatchSize+x][0] = 
                    (ref_patch_with_boarder[ (y+1)*step+(x+1)+1 ] - 
                     ref_patch_with_boarder[ (y+1)*step+x] )/255.0;
                _fej[y*WarpPatchSize+x][1] = 
                    (ref_patch_with_boarder[ (y+2)*step+(x+1)   ] - 
                     ref_patch_with_boarder[ (y)*step+(x+1)     ] )/255.0;
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
            for ( int i=0; i<WarpPatchSize*WarpPatchSize; i++ )
            {
                if ( _use_fej )
                {
                    jacobians[0][i*2]   = _fej[i][0];
                    jacobians[0][i*2+1] = _fej[i][1];
                }
                else 
                    jacobians[0][i*2] = jacobians[0][i*2+1] = 0;
            }
        }
        
        bool visible = _enable && curr_x>(WarpHalfPatchSize+1) && curr_y>(WarpHalfPatchSize+1) 
            && curr_x<(_curr_img.cols-WarpHalfPatchSize-1) && curr_y<(_curr_img.rows-WarpHalfPatchSize-1); 

        // 超出范围，设残差为零，雅可比保持不变
        if ( visible == false )
        {
            // set all residuals to zero 
            for( int i=0; i<WarpPatchSize*WarpPatchSize; i++ )
                residuals[i] = 0;
            return true;
        }
        
        // compute the residuals 
        for ( int x=0; x<WarpPatchSize; x++ )
            for ( int y=0; y<WarpPatchSize; y++ ) 
            {
                double u = curr_x + x - WarpHalfPatchSize;
                double v = curr_y + y - WarpHalfPatchSize;
                uchar gray = cvutils::GetBilateralInterpUchar(u,v,_curr_img);
                residuals[y*WarpPatchSize+x] = 
                    (double(gray) - _ref_patch[ y*WarpPatchSize + x ])/255.0;
                    
                if ( setJacobian && _use_fej==false ) {
                    // 不用FEJ的话，就在current上面计算jacobian
                    double du = (cvutils::GetBilateralInterp(u+1, v, _curr_img) - cvutils::GetBilateralInterp(u-1,v,_curr_img))/2.0;
                    double dv = (cvutils::GetBilateralInterp(u, v+1, _curr_img) - cvutils::GetBilateralInterp(u,v-1,_curr_img))/2.0;
                    jacobians[0][(y*WarpPatchSize+x)*2 +0] += du;
                    jacobians[0][(y*WarpPatchSize+x)*2 +1] += dv;
                }
            } 
        return true;
    }
    
private:
    uint8_t* _ref_patch=nullptr; 
    const Mat& _curr_img; 
    double _fej[WarpPatchSize*WarpPatchSize][2];
    bool _use_fej = true;
    bool _enable = true; 
};
    
}

#endif