#include "ygz/optimizer.h"

// g2o 
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

using namespace g2o;

namespace ygz
{

namespace opti
{
    
void TwoViewBA ( 
    const long unsigned int& frameID1, 
    const long unsigned int& frameID2 )
{
    // set up g2o 
    BlockSolver_6_3::LinearSolverType* linearSolver = new LinearSolverCSparse<BlockSolver_6_3::PoseMatrixType>();
    BlockSolver_6_3* solver_ptr = new BlockSolver_6_3( linearSolver ); 
    OptimizationAlgorithmLevenberg* solver = new OptimizationAlgorithmLevenberg( solver_ptr ); 
    solver->setMaxTrialsAfterFailure(5);
    g2o::SparseOptimizer optimizer; 
    optimizer.setAlgorithm( solver );
    
    

}

}
}
