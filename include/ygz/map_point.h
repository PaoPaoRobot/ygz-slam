#ifndef MAP_POINT_H_
#define MAP_POINT_H_

#include "ygz/common_include.h"

namespace ygz {
    
struct MapPoint {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    typedef shared_ptr<MapPoint> Ptr; 
    MapPoint()  {}
    unsigned long   _id =0; 
    Vector2d        _pos_pixel = Vector2d(0,0);
    Vector3d        _norm = Vector3d(0,0,0);  // normal 
    int             _pyramid_level =0; 
    unsigned long   _first_observed =0; 
    Vector3d        _pos_world =Vector3d(0,0,0); 
    map<unsigned long, size_t> _obs;   // observations, first=the frame, second=feature index
    bool            _bad =false; 
}; 
}

#endif 
