#ifndef FRAME_H_
#define FRAME_H_
#include "ygz/common_include.h"
#include "ygz/camera.h"
#include "ygz/map_point.h"

namespace ygz {
    
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
    
    // return the camera position in the world 
    inline Vector3d Pos() const { return _T_c_w.inverse().translation(); }
    
    // check whether a point is in frame 
    inline bool InFrame( const Vector2d& pixel, const int& boarder = 10 ) const {
        return pixel[0] >= boarder && pixel[0] < _color.cols - boarder 
            && pixel[1] >= boarder && pixel[1] < _color.rows - boarder;
    }
    
    inline void AddMapPoint( const unsigned long& id ) { _map_point.push_back(id); }
    
public:
    // data 
    unsigned long _id   =0; 
    double  _timestamp  =0; 
    SE3     _T_c_w      =SE3();       // pose 
    bool    _is_keyframe    =false; 
    
    // NOTE: 正式的map point是要放到memory里的，而特征提取过程中的那些只能是candidate
    list<unsigned long> _map_point; // associated map point 
    vector<MapPoint>    _map_point_candidates;  // candidates 
    
    // images 
    Mat     _color;     // if we have 
    Mat     _depth;     // if we have 
    
    // pyramid 
    static int _pyramid_level;  // pyramid 越大，图像越粗糙，
    vector<Mat>  _pyramid;      // gray image pyramid, it must be CV_8U
    
    // camera 
    static PinholeCamera::Ptr   _camera;
    
    // Grid 
    vector< vector<int> > _grid;        // grid occupancy 
    
protected:
    // inner functions 
    // build pyramid 是从fine到coarse的过程 
    void CreateImagePyramid();
    
};

}

#endif
