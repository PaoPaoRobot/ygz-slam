#ifndef MAP_POINT_H_
#define MAP_POINT_H_

#include "ygz/common_include.h"

namespace ygz {
    
class Frame; 

struct MapPoint {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    typedef shared_ptr<MapPoint> Ptr; 
    MapPoint()  {}
    
    unsigned long   _id =0; 
    Vector2d        _pos_tracked = Vector2d(0,0);
    Vector3d        _norm = Vector3d(0,0,0);  // normal 
    int             _pyramid_level =0; 
    unsigned long   _first_frame =0; 
    
    Vector3d        _pos_world =Vector3d(0,0,0); 
    map<Frame*, size_t>     _obs;   // observations 
    bool            _bad =true; 
    
}; 
}

#endif 
