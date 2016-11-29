#ifndef CAMERA_H_
#define CAMERA_H_

#include "ygz/common_include.h"
#include "ygz/config.h"

namespace ygz
{

class PinholeCamera
{
public:
    typedef shared_ptr<PinholeCamera> Ptr;
    PinholeCamera( ) {
        _fx = Config::get<float> ( "camera.fx" );
        _fy = Config::get<float> ( "camera.fy" );
        _cx = Config::get<float> ( "camera.cx" );
        _cy = Config::get<float> ( "camera.cy" );

        _k1 = Config::get<float> ( "camera.k1" );
        _k2 = Config::get<float> ( "camera.k2" );
        _p1 = Config::get<float> ( "camera.p1" );
        _p2 = Config::get<float> ( "camera.p2" );

        _f = ( _fx+_fy ) /2;
        _mat = ( cv::Mat_<double> ( 3,3 ) <<  _fx, 0, _cx, 0, _fy, _cy, 0, 0, 1 );
    }

    inline Eigen::Matrix3f GetCameraMatrix() const {
        Eigen::Matrix3f m;
        m <<    _fx, 0, _cx,
          0, _fy, _cy,
          0, 0, 1;
        return m;
    }

    inline Mat GetCameraMatrixCV() const {
        return _mat;
    }

    // coordinate transform: world, camera, pixel
    inline Vector3d World2Camera ( const Vector3d& p_w, const SE3& T_c_w ) {
        return T_c_w*p_w;
    }

    inline Vector3d Camera2World ( const Vector3d& p_c, const SE3& T_c_w ) {
        return T_c_w.inverse() *p_c;
    }

    inline Vector2d Camera2Pixel ( const Vector3d& p_c ) {
        return Vector2d (
                   _fx * p_c ( 0,0 ) / p_c ( 2,0 ) + _cx,
                   _fy * p_c ( 1,0 ) / p_c ( 2,0 ) + _cy
               );
    }
    
    // used in ceres 
    template<typename T>
    inline void Camera2Pixel( const T* const p_c, T* p_x ) {
        p_x[0] =  (T) _fx * p_c[0] / p_c[2] + _cx;
        p_x[1] =  (T) _fy * p_c[1] / p_c[2] + _cy;
    }

    inline Vector3d Pixel2Camera ( const Vector2d& p_p, double depth=1 ) {
        return Vector3d (
                   ( p_p ( 0,0 )-_cx ) *depth/_fx,
                   ( p_p ( 1,0 )-_cy ) *depth/_fy,
                   depth
               );
    }

    inline Vector2d Pixel2Camera2D ( const Vector2d& p_p ) {
        return Vector2d (
                   ( p_p[0] - _cx ) / _fx,
                   ( p_p[1] - _cy ) / _fy
               );
    }

    inline Vector3d Pixel2World ( const Vector2d& p_p, const SE3& T_c_w, double depth=1 ) {
        return Camera2World ( Pixel2Camera ( p_p, depth ), T_c_w );
    }

    Vector2d World2Pixel ( const Vector3d& p_w, const SE3& T_c_w ) {
        return Camera2Pixel ( World2Camera ( p_w, T_c_w ) );
    }

    // accessors
    inline float fx() const {
        return _fx;
    }
    inline float fy() const {
        return _fy;
    }
    inline float cx() const {
        return _cx;
    }
    inline float cy() const {
        return _cy;
    }
    inline float focal() const {
        return _f;
    }

    // undistort, input should be normalied 2d points
    inline Vector2d UndistortPoint ( const Vector2d pt ) {
        double r2 = pt[0]*pt[0] + pt[1]*pt[1];
        Vector2d v;
        v[0] = pt[0] * ( 1+_k1*r2+_k2*r2*r2 ) + 2*_p1*pt[0]*pt[1]+_p2* ( r2+2*pt[0] );
        v[1] = pt[1] * ( 1+_k1*r2+_k2*r2*r2 ) + 2*_p2*pt[0]*pt[1]+_p1* ( r2+2*pt[1] );
        return v;
    }

protected:
    // intrinsics
    float _fx, _fy, _cx, _cy, _f;
    // distortion
    float _k1, _k2, _p1, _p2;

    Mat _mat;
};


// TODO: think about RGBD camera

}

#endif
