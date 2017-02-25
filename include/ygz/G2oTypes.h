#ifndef G2O_TYPES_YGZ_H_
#define G2O_TYPES_YGZ_H_

#include "ygz/Basic/Common.h"
#include "ygz/Basic/Camera.h"

namespace ygz {
 
class PinholeCamera; 

// SE3 pose defined by Sophus::SE3, easy to use 
// pose is define as T_c_w, use left multiply 
class VertexSE3Sophus : public g2o::BaseVertex<6, Vector6d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    VertexSE3Sophus() : BaseVertex<6, Vector6d>() {} 
    virtual bool read( std::istream& is ) override {
      Vector6d est;
      for (int i=0; i<6; i++)
	is  >> est[i];
      setEstimate(est);
      return true;
    }
    
    virtual bool write( std::ostream& os ) const override { 
      for (int i=0; i<6; i++)
	os << _estimate[i] << " ";
      return os.good();
    }
    
    // set to identity
    virtual void setToOriginImpl() override {
        for ( int i=0; i<6; i++ ) 
            _estimate[i] = 0;
    }
    
    // left multiply
    virtual void oplusImpl( const double* update ) override {
        Vector6d v;
        v << update[3], update[4], update[5], update[0], update[1], update[2]; 
	Vector6d est;
	est<<_estimate[3], _estimate[4], _estimate[5], _estimate[0], _estimate[1], _estimate[2]; 
	Vector6d vec6d_tr = (SE3::exp(v) * SE3::exp(est)).log();
	_estimate<<vec6d_tr[3], vec6d_tr[4], vec6d_tr[5], vec6d_tr[0], vec6d_tr[1], vec6d_tr[2];
    }
    ~VertexSE3Sophus() = default;
};

// Edge of SE3 and PointXYZ 
class EdgeSophusSE3ProjectXYZ 
: public g2o::BaseBinaryEdge <2, Vector2d, 
    g2o::VertexSBAPointXYZ, VertexSE3Sophus> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  double fx, fy, cx, cy;
  
  EdgeSophusSE3ProjectXYZ() : BaseBinaryEdge<2, Vector2d, g2o::VertexSBAPointXYZ, VertexSE3Sophus>(){}
  
  void setCamera(const PinholeCamera* camera ) 
  {
    fx=camera->fx();
    fy=camera->fy();
    cx=camera->cx();
    cy=camera->cy();
  } 

  bool read(std::istream& is) override {
    return false;
  };

  bool write(std::ostream& os) const  override {
    for (int i=0; i<2; i++){
    os << measurement()[i] << " ";
    }

    for (int i=0; i<2; i++)
      for (int j=i; j<2; j++){
	os << " " <<  information()(i,j);
      }
    return os.good();
  };

  virtual void computeError()  {
    const VertexSE3Sophus* v1 = static_cast<const VertexSE3Sophus*>(_vertices[1]);  
    const g2o::VertexSBAPointXYZ* v2 = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
    Vector2d obs(_measurement);
    Vector6d est;
    est<<v1->estimate()[3], v1->estimate()[4], v1->estimate()[5], v1->estimate()[0], v1->estimate()[1], v1->estimate()[2]; 
    _error = obs - camProject(SE3::exp(est)*v2->estimate());
  }

  virtual void linearizeOplus()
  {
    VertexSE3Sophus * vj = static_cast<VertexSE3Sophus *>(_vertices[1]);
    g2o::VertexSBAPointXYZ* vi = static_cast<g2o::VertexSBAPointXYZ*>(_vertices[0]);
    Vector3d xyz = vi->estimate();
    Vector6d est;
    est<<vj->estimate()[3], vj->estimate()[4], vj->estimate()[5], vj->estimate()[0], vj->estimate()[1], vj->estimate()[2]; 
    Sophus::SE3 T(SE3::exp(est));
    Vector3d xyz_trans = SE3::exp(est)*xyz;

    double x = xyz_trans[0];
    double y = xyz_trans[1];
    double z = xyz_trans[2];
    double z_2 = z*z;

    Eigen::Matrix<double,2,3> tmp;
    tmp(0,0) = fx;
    tmp(0,1) = 0;
    tmp(0,2) = -x/z*fx;

    tmp(1,0) = 0;
    tmp(1,1) = fy;
    tmp(1,2) = -y/z*fy;

    _jacobianOplusXi =  -1./z * tmp * T.rotation_matrix();

    _jacobianOplusXj(0,0) =  x*y/z_2 *fx;
    _jacobianOplusXj(0,1) = -(1+(x*x/z_2)) *fx;
    _jacobianOplusXj(0,2) = y/z *fx;
    _jacobianOplusXj(0,3) = -1./z *fx;
    _jacobianOplusXj(0,4) = 0;
    _jacobianOplusXj(0,5) = x/z_2 *fx;

    _jacobianOplusXj(1,0) = (1+y*y/z_2) *fy;
    _jacobianOplusXj(1,1) = -x*y/z_2 *fy;
    _jacobianOplusXj(1,2) = -x/z *fy;
    _jacobianOplusXj(1,3) = 0;
    _jacobianOplusXj(1,4) = -1./z *fy;
    _jacobianOplusXj(1,5) = y/z_2 *fy;
  }

  Vector2d camProject(const Vector3d & trans_xyz)
  { 
    Vector2d proj;
    proj(0) = trans_xyz(0)/trans_xyz(2);
    proj(1) = trans_xyz(1)/trans_xyz(2);
    
    Vector2d res;
    res(0) = proj(0)*fx + cx;
    res(1) = proj(1)*fy + cy;
    return res;
  }
 
};

class EdgeSophusSE3ProjectXYZOnlyPose: public g2o::BaseUnaryEdge<2, Vector2d, VertexSE3Sophus>{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  double fx, fy, cx, cy;
  Vector3d Xw;
  
  EdgeSophusSE3ProjectXYZOnlyPose(){}
  
  void setCamera(const PinholeCamera* camera ) 
  {
    fx=camera->fx();
    fy=camera->fy();
    cx=camera->cx();
    cy=camera->cy();
  } 
  

  bool read(std::istream& is) override {return false;};

  bool write(std::ostream& os) const override {return false;};

  void computeError()  override{
    const VertexSE3Sophus* v1 = static_cast<const VertexSE3Sophus*>(_vertices[0]);
    Vector2d obs(_measurement);
    Vector6d est;
    est<<v1->estimate()[3], v1->estimate()[4], v1->estimate()[5], v1->estimate()[0], v1->estimate()[1], v1->estimate()[2]; 
    _error = obs - camProject(SE3::exp(est)*Xw);   
  }

  virtual void linearizeOplus() override{
    VertexSE3Sophus * v1 = static_cast<VertexSE3Sophus *>(_vertices[0]);
    Vector6d est;
    est<<v1->estimate()[3], v1->estimate()[4], v1->estimate()[5], v1->estimate()[0], v1->estimate()[1], v1->estimate()[2]; 
    Sophus::SE3 T(SE3::exp(est));
    Vector3d xyz_trans = T.rotation_matrix()*Xw + T.translation();

    double x = xyz_trans(0);
    double y = xyz_trans(1);
    double invz = 1.0/xyz_trans(2);
    double invz_2 = invz*invz;

    _jacobianOplusXi(0,0) =  x*y*invz_2 *fx;
    _jacobianOplusXi(0,1) = -(1+(x*x*invz_2)) *fx;
    _jacobianOplusXi(0,2) = y*invz *fx;
    _jacobianOplusXi(0,3) = -invz *fx;
    _jacobianOplusXi(0,4) = 0;
    _jacobianOplusXi(0,5) = x*invz_2 *fx;

    _jacobianOplusXi(1,0) = (1+y*y*invz_2) *fy;
    _jacobianOplusXi(1,1) = -x*y*invz_2 *fy;
    _jacobianOplusXi(1,2) = -x*invz *fy;
    _jacobianOplusXi(1,3) = 0;
    _jacobianOplusXi(1,4) = -invz *fy;
    _jacobianOplusXi(1,5) = y*invz_2 *fy;
  }

  Vector2d camProject(const Vector3d & trans_xyz)
  {
    Vector2d proj;
    proj(0) = trans_xyz(0)/trans_xyz(2);
    proj(1) = trans_xyz(1)/trans_xyz(2);
    
    Vector2d res;
    res(0) = proj(0)*fx + cx;
    res(1) = proj(1)*fy + cy;
    return res;
  }
  
};

}

#endif 