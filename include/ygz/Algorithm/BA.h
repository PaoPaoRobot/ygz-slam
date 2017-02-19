#ifndef YGZ_BA_H_
#define YGZ_BA_H_

#include "ygz/Basic.h"
// Bundle Adjustment相关算法，在VO和后端都有调用

namespace ygz 
{
    
namespace ba
{
    
// 初始化中两个帧之间的BA, 第一个帧是固定不动的
/**
 * @brief Two view bundle adjustment, used in intialization 
 * @param[in] ref the reference frame's pose 
 * @param[out] curr the current frame's pose, should have initial value provided by initialization, will be corrected after BA
 * @param[in] px_ref 2D features in reference 
 * @param[in] px_curr 2D features in current, should be matched one by one with px_ref 
 * @param[out] inlier true if it is a inlier, should have initial values provided by initializer  
 * @param[out] pts_ref the triangluated points in ref, should have initial values, will be corrected by BA. 
 */
void TwoViewBACeres( 
    const SE3& ref,
    SE3& curr, 
    const vector<Vector2d> px_ref, 
    const vector<Vector2d> px_curr, 
    vector<bool>& inlier,
    vector<Vector3d>& pts_ref
);



}

};

#endif // YGZ_BA_H_