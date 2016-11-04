#ifndef FRAME_H_
#define FRAME_H_
#include "ygz/common_include.h"
#include "ygz/camera.h"
#include "ygz/map_point.h"

namespace ygz {
    
struct Frame {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    Frame();
    Frame( const Frame& frame );

    static void SetCamera( PinholeCamera::Ptr camera ) {
        _camera = camera; 
    }
    
    unsigned long _id   =0; 
    double  _timestamp  =0; 
    SE3     _T_c_w      =SE3(); 
    bool    _is_keyframe    =false; 
    list<unsigned long> _map_point; // associated map point 
    
    // images 
    Mat     _color;     // if we have 
    vector<Mat>  _pyramid;      // gray image pyramid, it must be CV_32F
    
    static PinholeCamera::Ptr   _camera;
};

}

#endif
