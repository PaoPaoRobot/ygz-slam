// g2o 
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

#include "ygz/optimizer.h"
#include "ygz/g2o_types.h"
#include "ygz/memory.h"

using namespace g2o;

namespace ygz
{

namespace opti
{
    
void TwoViewBA ( 
    const long unsigned int& frameID1, 
    const long unsigned int& frameID2 )
{
    assert( Memory::GetFrame(frameID1) != nullptr && Memory::GetFrame(frameID2) != nullptr );
    
    // set up g2o 
    BlockSolver_6_3::LinearSolverType* linearSolver = new LinearSolverCSparse<BlockSolver_6_3::PoseMatrixType>();
    BlockSolver_6_3* solver_ptr = new BlockSolver_6_3( linearSolver ); 
    OptimizationAlgorithmLevenberg* solver = new OptimizationAlgorithmLevenberg( solver_ptr ); 
    solver->setMaxTrialsAfterFailure(5);
    g2o::SparseOptimizer optimizer; 
    optimizer.setAlgorithm( solver );
    
    // vertecies and edges 
    VertexSE3Sophus* v1 = new VertexSE3Sophus(); 
    v1->setId(0);
    Frame::Ptr frame1 = Memory::GetFrame( frameID1 );
    v1->setEstimate( frame1->_T_c_w );
    
    VertexSE3Sophus* v2 = new VertexSE3Sophus(); 
    v2->setId(1);
    Frame::Ptr frame2 = Memory::GetFrame( frameID2 );
    v2->setEstimate( frame2->_T_c_w );
    
    optimizer.addVertex( v1 );
    optimizer.addVertex( v2 );
    
    v1->setFixed(true); // fix the first one 
    
    // points and edges 
    map<unsigned long, VertexSBAPointXYZ*> vertex_points; 
    vector<EdgeSophusSE3ProjectXYZ*> edges; 
    
    frame1->_map_point;
    int pts_id = 2; 
    for ( auto iter = frame1->_map_point.begin(); iter!=frame1->_map_point.size(); iter++ ) {
        MapPoint::Ptr map_point = Memory::GetMapPoint( *iter );
        if ( map_point==nullptr && map_point->_bad==true )
            continue; 
        VertexSBAPointXYZ* pt_xyz = new VertexSBAPointXYZ();
        pt_xyz->setId( pts_id++ );
        pt_xyz->setEstimate( map_point->_pos_world );
        optimizer.addVertex( pt_xyz );
        vertex_points[map_point->_id] = pt_xyz;
        
        EdgeSophusSE3ProjectXYZ* edge1 = new EdgeSophusSE3ProjectXYZ();
        edge1->setVertex( 0, pt_xyz );
        edge1->setVertex( 1, v1 );
        edge1->setMeasurement( frame1->_camera->Pixel2Camera2D( map_point->_obs[frameID1] ) );
        edge1->setInformation( Eigen::Matrix2d::Identity() );
        // robust kernel ? 
        optimizer.addEdge( edge1 );
        edges.push_back( edge1 );
        
        EdgeSophusSE3ProjectXYZ* edge2 = new EdgeSophusSE3ProjectXYZ();
        edge2->setVertex( 0, pt_xyz );
        edge2->setVertex( 1, v2 );
        edge2->setMeasurement( frame2->_camera->Pixel2Camera2D( map_point->_obs[frameID2] ) );
        edge2->setInformation( Eigen::Matrix2d::Identity() );
        // robust kernel ? 
        optimizer.addEdge( edge2 );
        edges.push_back( edge2 );
    }
    
    // do optimization!  >_<
    optimizer.initializeOptimization(); 
    optimizer.computeActiveErrors();
    LOG(INFO) << "initial error: " << optimizer.activeChi2() << endl;
    optimizer.optimize( 5 );
    optimizer.computeActiveErrors();
    LOG(INFO) << "optimized error: " << optimizer.activeChi2() << endl;
    
    // update the key-frame and map points 
    // TODO delete the outlier! 但是outlier应该在前面的估计中去过一次了啊 
    frame1->_T_c_w = v1->estimate();
    frame2->_T_c_w = v2->estimate();

    for ( auto v:vertex_points ) {
        MapPoint::Ptr map_point = Memory::GetMapPoint( v.first );
        map_point->_pos_world = v.second->estimate();
    }
}

}
}
