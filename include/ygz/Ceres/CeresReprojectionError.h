#ifndef YGZ_CERES_REPROJECTION_ERROR_H_
#define YGZ_CERES_REPROJECTION_ERROR_H_

#include <ceres/rotation.h>

#include "ygz/Basic.h"
#include "ygz/Algorithm/CVUtils.h"

namespace ygz 
{
    
// 重投影误差，给定归一化坐标，这样可以省掉内参部分的计算
// 这东西自己写jacobian简直就是作死，没法自定义更新步骤就意味着雅可比里有SE3上的那个花体J，巨复杂
// 所以干脆自动求导就完了，省心省力，不知道效率会不会低一些
class CeresReprojectionError
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW 
    CeresReprojectionError( const Vector2d& pt_cam ): _pt_cam(pt_cam) { }
    
    void SetWeight( const double& weight )
    {
        _weight = weight;
    }
    
    void Enable( bool enable=true ) 
    {
        _enable = true; 
    }
    
    // Error = z-(Rp+t)
    // pose is [so3,t], TCW  
    template< typename T >
    bool operator() (
        const T* const pose, 
        const T* const pt_world,
        T* residuals
    ) const
    {
        if ( _enable == false )
        {
            residuals[0] = residuals[1] = T(0);
            return true;
        }
        
        T p[3];
        T rot[3];
        for ( size_t i=0; i<3; i++ )
            rot[i] = pose[i+3];
        ceres::AngleAxisRotatePoint<T>( rot, pt_world, p );
        
        p[0] += pose[0]; 
        p[1] += pose[1]; 
        p[2] += pose[2]; 
        
        /*
        if ( p[2] < T(0) ) {
            // 这个必须 check 否则点容易被优化到负深度上去
            residuals[0] = residuals[1] = T(0);
            LOG(INFO)<<"invalid depth "<<p[2]<<endl;
            return false;
        }
        */
        
        residuals[0] = _weight* (_pt_cam[0] - p[0]/p[2]);
        residuals[1] = _weight* (_pt_cam[1] - p[1]/p[2]);
        
        return true; 
    }
    
private:
    Vector2d _pt_cam; // observation: normalized camera coordinate 
    double _weight = 1.0; // 权重
    bool _enable =true;   // 使能
};

}
#endif // YGZ_CERES_REPROJECTION_ERROR_H_
