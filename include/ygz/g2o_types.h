#ifndef G2O_TYPES_YGZ_H_
#define G2O_TYPES_YGZ_H_

#include <ygz/common_include.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

namespace ygz {
    
// SE3 pose defined by Sophus::SE3, easy to use 
// pose is define as T_c_w, use left multiply 
class VertexSE3Sophus : public g2o::BaseVertex<6, Vector6d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    VertexSE3Sophus() {} 
    virtual bool read( std::istream& is ) override { return false; }
    virtual bool write( std::ostream& os ) const override { return false; }
    virtual void setToOriginImpl() override {
        for ( int i=0; i<6; i++ ) 
            _estimate[i] = 0;
    }
    
    // left multiply
    virtual void oplusImpl( const double* update ) {
        Vector6d v;
        v << update[0], update[1], update[2], update[3], update[4], update[5]; 
        _estimate = (SE3::exp(v) * SE3::exp(_estimate)).log();
    }
};

// Edge of SE3 and PointXYZ 
// note that in the point we use normalized coordinates 
class EdgeSophusSE3ProjectXYZ 
: public g2o::BaseBinaryEdge <2, Vector2d, 
    g2o::VertexSBAPointXYZ, VertexSE3Sophus> {
public: 
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    virtual bool read( std::istream& is ) override { return false; }
    virtual bool write( std::ostream& os ) const override { return false; }
    
    // e = u - 1/z * (Rp+t)
    void computeError() override {
        const VertexSE3Sophus* v1 = static_cast<const VertexSE3Sophus*>(_vertices[1]);
        const g2o::VertexSBAPointXYZ* v2 = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
        Vector3d pt = SE3::exp(v1->estimate())*v2->estimate();
        Vector2d ptn( pt[0]/pt[2], pt[1]/pt[2] );
        _error = _measurement - ptn;
    }
    
    virtual void linearizeOplus() override {
        const VertexSE3Sophus* v1 = static_cast<const VertexSE3Sophus*>(_vertices[1]);
        const g2o::VertexSBAPointXYZ* v2 = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
        Vector3d pt = SE3::exp(v1->estimate())*v2->estimate();
        
        double x = pt[0];
        double y = pt[1];
        double z = pt[2];
        double z_inv = 1./z; 
        double z_inv_2 = z_inv*z_inv;
        
        Eigen::Matrix<double,2,3> tmp; 
        tmp(0,0) = z_inv;
        tmp(0,1) = 0;
        tmp(0,2) = -x*z_inv_2;
        
        tmp(1,0) = 0;
        tmp(1,1) = z_inv;
        tmp(1,2) = -y*z_inv_2;
        
        _jacobianOplusXi = - tmp * SE3::exp(v1->estimate()).rotation_matrix();
        
        _jacobianOplusXj(0,0) = -z_inv;              // -1/z
        _jacobianOplusXj(0,1) = 0.0;                 // 0
        _jacobianOplusXj(0,2) = x*z_inv_2;           // x/z^2
        _jacobianOplusXj(0,3) = y*_jacobianOplusXj(0,2);            // x*y/z^2
        _jacobianOplusXj(0,4) = -(1.0 + x*_jacobianOplusXj(0,2));   // -(1.0 + x^2/z^2)
        _jacobianOplusXj(0,5) = y*z_inv;             // y/z
        
        _jacobianOplusXj(1,0) = 0.0;                 // 0
        _jacobianOplusXj(1,1) = -z_inv;              // -1/z
        _jacobianOplusXj(1,2) = y*z_inv_2;           // y/z^2
        _jacobianOplusXj(1,3) = 1.0 + y*_jacobianOplusXj(1,2);      // 1.0 + y^2/z^2
        _jacobianOplusXj(1,4) = -_jacobianOplusXj(0,3);             // -x*y/z^2
        _jacobianOplusXj(1,5) = -x*z_inv;            // x/z
    }
};



}

#endif 