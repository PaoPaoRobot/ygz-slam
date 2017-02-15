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
    
    for ( size_t i=0; i<px_ref; i++ ) 
    {
        auto error1 = new CeresReprojectionErrorPointOnly( cam->Pixel2Camera2D(px_ref[i]), ref );
        // ref frame, point only, pose is fixed 
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<CeresReprojectionErrorPointOnly, 2,3> ( error1 ), 
            nullptr, 
            pts_ref[i].data()
        );
        // curr frame, both point and pose 
        auto error2 = new CeresReprojectionError( px_curr[i] );
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<CeresReprojectionError, 2,6,3>
        )
    }
    

}

    
    
    
}
}