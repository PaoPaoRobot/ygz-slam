#include "ygz/Basic.h"
#include "ygz/Algorithm/CVUtils.h"
#include "ygz/CeresTypes.h"

namespace ygz 
{
    
namespace cvutils 
{
   
bool Align2DCeres(
    const Mat& cur_img, uint8_t* ref_patch, 
    Vector2d& cur_px_estimate)
{
    ceres::Problem problem;
    Vector2d px = cur_px_estimate;
    CeresAlignmentError* p = new CeresAlignmentError(
        ref_patch, cur_img, false
    );
    
    problem.AddResidualBlock(
        p, 
        nullptr,
        cur_px_estimate.data()
    );
    
    ceres::Solver::Options options;
    options.max_num_iterations = 10;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    ceres::Solver::Summary summary;
    ceres::Solve( options, &problem, &summary );
    
    // LOG(INFO) << summary.final_cost << endl;
    // 判断结果是否合理
    bool bad = false;
    if ( (px-cur_px_estimate).norm() > 5 )
        bad = true;
    if ( summary.final_cost > 2000 ) 
        bad = true;
    if ( bad == false ) 
        return true; 
    
    // LOG(INFO)<<"retrying"<<endl;
    // 尝试不使用FeJ重新算一遍
    
    /*
    cur_px_estimate = px; 
    p->SetFej( false );
    ceres::Solve( options, &problem, &summary );
    // LOG(INFO) << summary.final_cost << endl;
    
    if ( (px-cur_px_estimate).norm() > 5 )
        return false;
    if ( summary.final_cost > 2000 ) 
        return false;
    return true;
    */
    return false;
}

    
}
    
}