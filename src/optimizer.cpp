// g2o 
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

#include <opencv2/imgproc/imgproc.hpp>

#include "ygz/common_include.h"
#include "ygz/optimizer.h"
#include "ygz/g2o_types.h"
#include "ygz/ceres_types.h"
#include "ygz/memory.h"

using namespace g2o;

namespace ygz
{

namespace opti
{
    
void TwoViewBAG2O ( 
    const long unsigned int& frameID1, 
    const long unsigned int& frameID2 
)
{
    assert( Memory::GetFrame(frameID1) != nullptr && Memory::GetFrame(frameID2) != nullptr );
    
    // set up g2o 
    typedef g2o::BlockSolver_6_3 Block;
    Block::LinearSolverType* linearSolver = new LinearSolverEigen<Block::PoseMatrixType>();
    Block* solver_ptr = new Block( linearSolver ); 
    OptimizationAlgorithmLevenberg* solver = new OptimizationAlgorithmLevenberg( solver_ptr ); 
    g2o::SparseOptimizer optimizer; 
    optimizer.setAlgorithm( solver );
    
    // vertecies and edges 
    VertexSE3Sophus* v1 = new VertexSE3Sophus(); 
    v1->setId(0);
    Frame::Ptr frame1 = Memory::GetFrame( frameID1 );
    v1->setEstimate( frame1->_T_c_w.log() );
    
    VertexSE3Sophus* v2 = new VertexSE3Sophus(); 
    v2->setId(1);
    Frame::Ptr frame2 = Memory::GetFrame( frameID2 );
    v2->setEstimate( frame2->_T_c_w.log() );
    
    optimizer.addVertex( v1 );
    optimizer.addVertex( v2 );
    optimizer.setVerbose( false );
    
    v1->setFixed(true); // fix the first one 
    
    // points and edges 
    map<unsigned long, VertexSBAPointXYZ*> vertex_points; 
    vector<EdgeSophusSE3ProjectXYZ*> edges; 
    int pts_id = 2; 
    
    /** * debug only 
    // print all related data
    for ( auto iter = frame1->_map_point.begin(); iter!=frame1->_map_point.end(); iter++ ) {
        MapPoint::Ptr map_point = Memory::GetMapPoint( *iter );
        map_point->PrintInfo();
        LOG(INFO) << "observed in frame 1: "<< map_point->_obs[frameID1] << endl;
        LOG(INFO) << "observed in frame 2: "<< map_point->_obs[frameID2] << endl;
    }
    
    LOG(INFO) << endl;
    **/
    
    
    for ( auto iter = frame1->_map_point.begin(); iter!=frame1->_map_point.end(); iter++ ) {
        MapPoint::Ptr map_point = Memory::GetMapPoint( *iter );
        if ( map_point==nullptr && map_point->_bad==true )
            continue; 
        
        VertexSBAPointXYZ* pt_xyz = new VertexSBAPointXYZ();
        pt_xyz->setId( pts_id++ );
        pt_xyz->setEstimate( map_point->_pos_world );
        pt_xyz->setMarginalized( true );
        optimizer.addVertex( pt_xyz );
        vertex_points[map_point->_id] = pt_xyz;
        
        EdgeSophusSE3ProjectXYZ* edge1 = new EdgeSophusSE3ProjectXYZ();
        edge1->setVertex( 0, pt_xyz );
        edge1->setVertex( 1, v1 );
        
        edge1->setMeasurement( frame1->_camera->Pixel2Camera2D( map_point->_obs[frameID1].head<2>() ) );
        edge1->setInformation( Eigen::Matrix2d::Identity() );
        // robust kernel ? 
        optimizer.addEdge( edge1 );
        edges.push_back( edge1 );
        
        EdgeSophusSE3ProjectXYZ* edge2 = new EdgeSophusSE3ProjectXYZ();
        edge2->setVertex( 0, pt_xyz );
        edge2->setVertex( 1, v2 );
        edge2->setMeasurement( frame2->_camera->Pixel2Camera2D( map_point->_obs[frameID2].head<2>() ) );
        edge2->setInformation( Eigen::Matrix2d::Identity() );
        // robust kernel ? 
        optimizer.addEdge( edge2 );
        edges.push_back( edge2 );
    }
    
    LOG(INFO) << "edges: "<<edges.size() <<endl;
    
    // do optimization!  >_<
    optimizer.initializeOptimization(); 
    // optimizer.computeActiveErrors();
    // LOG(INFO) << "initial error: " << optimizer.activeChi2() << endl;
    optimizer.optimize( 10 );
    // optimizer.computeActiveErrors();
    // LOG(INFO) << "optimized error: " << optimizer.activeChi2() << endl;
    
    // update the key-frame and map points 
    // TODO delete the outlier! 但是outlier应该在前面的估计中去过一次了啊 
    // LOG(INFO) << "frame 2 before optimization: \n" << frame2->_T_c_w.matrix()<<endl;
    frame1->_T_c_w = SE3::exp(v1->estimate());
    frame2->_T_c_w = SE3::exp(v2->estimate());
    // LOG(INFO) << "frame 2 after optimization: \n" << frame2->_T_c_w.matrix()<<endl;

    for ( auto v:vertex_points ) {
        MapPoint::Ptr map_point = Memory::GetMapPoint( v.first );
        map_point->_pos_world = v.second->estimate();
    }
}

void TwoViewBACeres ( 
    const long unsigned int& frameID1, 
    const long unsigned int& frameID2 
)
{
    assert( Memory::GetFrame(frameID1) != nullptr && Memory::GetFrame(frameID2) != nullptr );
    
    Frame::Ptr frame1 = Memory::GetFrame( frameID1 );
    Frame::Ptr frame2 = Memory::GetFrame( frameID2 );
    
    Vector6d pose1, pose2;
    Vector3d r1 = frame1->_T_c_w.so3().log(), t1=frame1->_T_c_w.translation();
    Vector3d r2 = frame2->_T_c_w.so3().log(), t2=frame2->_T_c_w.translation();
    pose1.head<3>() = t1; 
    pose1.tail<3>() = r1; 
    pose2.head<3>() = t2; 
    pose2.tail<3>() = r2; 
    
    ceres::Problem problem; 
    for ( auto iter = frame1->_map_point.begin(); iter!= frame1->_map_point.end(); iter++ ) {
        MapPoint::Ptr map_point = Memory::GetMapPoint( *iter );
        for ( auto obs:map_point->_obs ) {
            Vector2d px = frame1->_camera->Pixel2Camera2D( obs.second.head<2>() ); 
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<CeresReprojectionError,2,6,3> (
                    new CeresReprojectionError(px)
                ), 
                nullptr, 
                obs.first==frame1->_id ? pose1.data() : pose2.data(), 
                map_point->_pos_world.data()
            );
        }
    }
    
    ceres::Solver::Options options; 
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true; 
    
    ceres::Solver::Summary summary;
    ceres::Solve( options, &problem, &summary );
    cout<< summary.FullReport() << endl;
    
    // set the value of two frames 
    frame1->_T_c_w = SE3(
        SO3::exp( pose1.tail<3>() ), pose1.head<3>()
    );
    
    frame2->_T_c_w = SE3(
        SO3::exp( pose2.tail<3>() ), pose2.head<3>()
    );
    
}

void SparseImgAlign::SparseImageAlignmentCeres ( 
    Frame::Ptr frame1, Frame::Ptr frame2,
    const int& pyramid_level
)
{
    _pyramid_level = pyramid_level;      
    _scale = 1<<pyramid_level;
    if ( frame1 == _frame1 ) {
        // 没必要重新算 ref 的 patch
        _have_ref_patch = true;
    }
    
    _frame1 = frame1;
    _frame2 = frame2;
    
    cv::Mat& curr_img = _frame2->_pyramid[pyramid_level];
    if ( _have_ref_patch==false ) {
        PrecomputeReferencePatches();
        LOG(INFO)<<"ref patterns: "<<_patterns_ref.size()<<endl;
    }
    
    // solve this problem 
    ceres::Problem problem;
    Vector6d pose2; 
    Vector3d r2 = _TCR.so3().log(), t2=_TCR.translation();
    pose2.head<3>() = t2; 
    pose2.tail<3>() = r2; 
    
    int index = 0;
    for ( auto it=_frame1->_map_point.begin(); it!=_frame1->_map_point.end(); it++, index++ ) {
        
        if (_visible_pts[index] == false) 
            continue;
        MapPoint::Ptr mappoint = Memory::GetMapPoint( *it );
        
        // camera coordinates in ref 
        Vector3d xyz_ref = _frame1->_camera->World2Camera( mappoint->_pos_world, _frame1->_T_c_w);
        
        /*
        LOG(INFO) << "index = "<<index<<endl;
        LOG(INFO) << "pattern ref = ";
        for ( int k=0; k<8; k++ ) {
            LOG(INFO) << _patterns_ref[index].pattern[k] <<" ";
        }
        LOG(INFO)<<endl;
        */
        
        problem.AddResidualBlock(
            new CeresReprojSparseDirectError( 
                    _frame2->_pyramid[_pyramid_level], 
                    _patterns_ref[index],
                    xyz_ref,
                    _frame1->_camera,
                    _scale
            ),
            // new ceres::HuberLoss(1), // TODO do I need Loss Function?
            nullptr, 
            pose2.data()
        );
    }
    
    ceres::Solver::Options options; 
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true; 
    
    ceres::Solver::Summary summary;
    ceres::Solve( options, &problem, &summary );
    cout<< summary.FullReport() << endl;
    
    
    // show the estimated pose 
    _TCR = SE3 (
        SO3::exp( pose2.tail<3>() ), 
        pose2.head<3>()
    );
    
    cv::Mat ref_show, curr_show; 
    cv::cvtColor( _frame1->_pyramid[pyramid_level], ref_show, CV_GRAY2BGR );
    cv::cvtColor( _frame2->_pyramid[pyramid_level], curr_show, CV_GRAY2BGR );
    
#ifdef DEBUG_VIZ 
    LOG(INFO) << "TCR = " << _TCR.matrix() << endl;
    index=0;
    for ( auto it=_frame1->_map_point.begin(); it!=_frame1->_map_point.end(); it++,index++ ) {
        if (_visible_pts[index] == false) 
            continue;
        MapPoint::Ptr mappoint = Memory::GetMapPoint( *it );
        // camera coordinates in ref 
        Vector3d xyz_ref = _frame1->_camera->World2Camera( mappoint->_pos_world, _frame1->_T_c_w);
        Vector2d px_ref = _frame1->_camera->Camera2Pixel( xyz_ref ) / _scale;
        // in current 
        Vector3d xyz_curr = _TCR * xyz_ref; 
        Vector2d px_curr = _frame2->_camera->Camera2Pixel( xyz_curr ) / _scale;
        
        cv::circle( ref_show, cv::Point2d(px_ref[0], px_ref[1]), 3, cv::Scalar(0,250,0));
        cv::circle( curr_show, cv::Point2d(px_curr[0], px_curr[1]), 3, cv::Scalar(0,250,0));
    }
    
    cv::imshow("ref", ref_show );
    cv::imshow("curr", curr_show );
    cv::waitKey(1);
#endif 
    
}


void SparseImgAlign::PrecomputeReferencePatches()
{
    LOG(INFO) << "frame 1 map points: "<<_frame1->_map_point.size()<<endl;
    _patterns_ref.clear();
    _patterns_ref.resize( _frame1->_map_point.size() );
    
    cv::Mat& ref_img = _frame1->_pyramid[_pyramid_level];
    _visible_pts = vector<bool>( _frame1->_map_point.size(), false);
    int i=0; 
    
    for ( auto it=_frame1->_map_point.begin(); it!=_frame1->_map_point.end(); it++, i++ ) {
        MapPoint::Ptr mappoint = Memory::GetMapPoint( *it );
        // camera coordinates in ref 
        Vector3d xyz_ref = _frame1->_camera->World2Camera( mappoint->_pos_world, _frame1->_T_c_w);
        Vector2d pixel_ref = _frame1->_camera->Camera2Pixel(xyz_ref)/_scale;
        
        if ( !_frame1->InFrame( pixel_ref, 10) )  // 不在图像范围中
            continue;
        _visible_pts[i] = true; 
        
        PixelPattern pattern_ref;
        for ( int k=0; k<PATTERN_SIZE; k++ ) {
            double u = pixel_ref[0] + PATTERN_DX[k];
            double v = pixel_ref[1] + PATTERN_DX[k];
            pattern_ref.pattern[k] = utils::GetBilateralInterp(u,v,ref_img);
        }
        _patterns_ref[i] = pattern_ref;
    }
    _have_ref_patch = true; 
}

void FrameToMapBAPoseOnly ( Frame::Ptr current, list< MatchPointCandidate >& candidates )
{

}



}
}
