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
    Vector3d        _norm = Vector3d(0,0,0);  // normal 
    int             _pyramid_level =0; 
    Vector3d        _pos_world =Vector3d(0,0,0); 
    map<unsigned long, Vector3d> _obs;   // observations, first=frame ID, second=(pixel coordinate, depth), depth by default is 1 
    bool            _bad =false; 
    
    // ORB feature 
    cv::Mat         _descriptor;        // 描述子
    cv::KeyPoint    _keypoint;          // 关键点
    
public:
    
    // 获得某个观测的相机坐标值
    Vector3d GetObservedPt( const unsigned long& keyframe_id );
    
    // for debug use
    void PrintInfo() {
        LOG(INFO) << "map point " << _id << "\nworld pos = " << _pos_world.transpose()<<endl;
        LOG(INFO) << "observations: "<<endl;
        for ( auto obs: _obs ) {
            LOG(INFO) << "from frame "<<obs.first << ", pixel pos = " << obs.second.transpose() << endl;
        }
    }
}; 
}

#endif 
