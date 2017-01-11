#ifndef YGZ_CERES_REPROJECTIONERROR_POSEONLY_H_
#define YGZ_CERES_REPROJECTIONERROR_POSEONLY_H_

#include <ceres/rotation.h>

#include "ygz/Basic.h"
#include "ygz/Algorithm/CVUtils.h"

namespace ygz 
{
    
class CeresReprojectionErrorPoseOnly
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW 
    CeresReprojectionErrorPoseOnly( 
        const Vector2d& pt_cam, const Vector3d& pt_world
    ): _pt_cam(pt_cam), _pt_world(pt_world)
    { }
    
    void SetEnable( bool enable = true ) 
    {
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
    double _weight = 1.0; // 权重
    bool _enable =true; 
};
    
}

#endif