#include "ygz/Basic.h"
#include "ygz/Algorithm/CVUtils.h"
#include "ygz/CeresTypes.h"

namespace ygz 
{
    
namespace cvutils 
{
   
bool Align2DCeres(
    const Mat& cur_img, 
    uint8_t* ref_patch, 
    uint8_t* ref_patch_with_boarder, 
    Vector2d& cur_px_estimate )
{
    ceres::Problem problem;
    Vector2d px = cur_px_estimate;
    CeresAlignmentError* p = new CeresAlignmentError(
        ref_patch, ref_patch_with_boarder, cur_img, true
    );
    
    problem.AddResidualBlock(
        p, 
        nullptr,
        cur_px_estimate.data()
    );
    
    // TODO compare the differenct options of ceres? 
    ceres::Solver::Options options;
    options.max_num_iterations = 10;
    // options.linear_solver_type = ceres::SPARSE_SCHUR;
    // options.minimizer_type = ceres::LINE_SEARCH;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    // options.num_threads = 2;
    // options.trust_region_strategy_type = ceres::DOGLEG;
    ceres::Solver::Summary summary;
    ceres::Solve( options, &problem, &summary );
    // LOG(INFO) << summary.FullReport()<<endl;
    
    // LOG(INFO) << summary.final_cost << endl;
    // 判断结果是否合理
    bool bad = false;
    if ( (px-cur_px_estimate).norm() > 5 )
        bad = true;
    if ( summary.final_cost > 2000 ) 
        bad = true;
    if ( bad == false ) 
        return true; 
    return false;
}

    
}
    
}