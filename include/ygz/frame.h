#ifndef FRAME_H_
#define FRAME_H_
#include "ygz/common_include.h"
#include "ygz/camera.h"
#include "ygz/map_point.h"

namespace ygz {
    
typedef shared_ptr<Frame> FramePtr;
    
struct Frame {
public:
    typedef shared_ptr<Frame> Ptr;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    // Frames should be created by memeory, otherwise their ID may not be valid 
    Frame() {} 
    Frame( const Frame& frame ) =delete;
    Frame( 
        const double& timestamp, 
        const SE3& T_c_w, 
        const bool is_keyframe, 
        const Mat& color, 
        const Mat& depth = Mat()
    ) : _timestamp(timestamp), _T_c_w(T_c_w), _is_keyframe(is_keyframe), _color(color), 
    _depth( depth ) {} 

    // called by system when reading parameters 
    static void SetCamera( PinholeCamera::Ptr camera ) {
        _camera = camera; 
    }
    
    // create the image pyramid, etc 
    void InitFrame();
    
    unsigned long _id   =0; 
    double  _timestamp  =0; 
    SE3     _T_c_w      =SE3(); 
    bool    _is_keyframe    =false; 
    list<unsigned long> _map_point; // associated map point 
    
    // images 
    Mat     _color;     // if we have 
    Mat     _depth;     // if we have 
    
    // pyramid 
    static int _pyramid_level; 
    vector<Mat>  _pyramid;      // gray image pyramid, it must be CV_8U
    
    // camera 
    static PinholeCamera::Ptr   _camera;
    
    // Grid 
    vector< vector<int> > _grid;        // grid occupancy 
    
protected:
    // inner functions 
    void CreateImagePyramid();
    
};

}

#endif
