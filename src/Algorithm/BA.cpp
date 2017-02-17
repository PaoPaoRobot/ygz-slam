#include "ygz/Algorithm/BA.h"
#include "ygz/CeresTypes.h"

namespace ygz 
{
namespace ba
{
    
void TwoViewBACeres(
    const SE3& ref, 
    SE3& curr, 
    const vector< Vector2d > px_ref, 
    const vector< Vector2d > px_curr, 
    vector< bool >& inlier, 
    vector< Vector3d >& pts_ref)
{
    assert( px_ref.size() == px_curr.size() );
    PinholeCamera* cam = Frame::GetCamera(); // must know camera intrinsics 
    assert( cam!=nullptr );
    
    Vector6d pose_curr; 
    pose_curr.head<3>() = curr.translation();
    pose_curr.tail<3>() = curr.so3().log();
    
    ceres::Problem problem;
    // add parameter blocks into ceres 
    vector<CeresReprojectionError*> errors_reproj; 
    vector<CeresReprojectionErrorPointOnly*> errors_reproj_point_only; 
    
    for ( size_t i=0; i<px_ref.size(); i++ ) 
    {
        auto error1 = new CeresReprojectionErrorPointOnly( cam->Pixel2Camera2D(px_ref[i]), ref );
        
        if ( inlier[i] == false )
            pts_ref[i] = Vector3d(0,0,1);
        
        // ref frame, point only, pose is fixed 
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<CeresReprojectionErrorPointOnly, 2,3> ( error1 ), 
            inlier[i]==true?nullptr:new ceres::HuberLoss(0.1),
            pts_ref[i].data()
        );
        
        // curr frame, both point and pose 
        auto error2 = new CeresReprojectionError( cam->Pixel2Camera2D( px_curr[i] ));
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<CeresReprojectionError,2,6,3>(
                error2 ), 
            inlier[i]==true?nullptr:new ceres::HuberLoss(0.1),
            pose_curr.data(), 
            pts_ref[i].data()
        );
    }
    
    // 膜拜 ORB 的四遍优化 ... 我就先做一遍吧，效果不好的话再说
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::DOGLEG;
    ceres::Solver::Summary summary;
    ceres::Solve( options, &problem, &summary );
    
    // update the current frame pose
    curr = SE3( SO3::exp(pose_curr.tail<3>()), pose_curr.head<3>() );
    
    // check the inliers 
    double ch2 = 5.991; // threshold for reprojection inliers 
    
    for ( size_t i=0; i<px_ref.size(); i++ )
    {
        Vector2d e1 = px_ref[i] - cam->World2Pixel( pts_ref[i], ref );
        Vector2d e2 = px_curr[i] - cam->World2Pixel( pts_ref[i], curr );
        double depth1 = cam->World2Camera( pts_ref[i], ref )[2];
        double depth2 = cam->World2Camera( pts_ref[i], curr )[2];
        
        if ( e1.dot(e1)>ch2 || e2.dot(e2)>ch2 || depth1<0 || depth2<0 )
        {
            inlier[i] = false;
        }
        else 
        {
            inlier[i] = true;
        }
    }
}

    
    
    
}
}