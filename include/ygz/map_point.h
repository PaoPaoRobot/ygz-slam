#ifndef MAP_POINT_H_
#define MAP_POINT_H_

#include "ygz/common_include.h"

namespace ygz {
    
// Map Point 代表一个地图点
// 它记录了自己在世界坐标系的位置，以及过去被观测到的情况
// 正式的map points要向memory注册以获得合法的id
// 在单目情况下，刚建立的map point没有深度，它必须依赖不断的观测后才会有正确的深度值
// 如果map point有了好的深度值，就可以用于直接法位姿估计以及image alignment，否则，我们需要在极线上进行搜索，才能得到正确的深度
    
struct MapPoint {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    typedef shared_ptr<MapPoint> Ptr; 
    MapPoint()  {}
    unsigned long   _id =0; 
    int             _pyramid_level =0;
    Vector3d        _pos_world =Vector3d(0,0,0); 
    unsigned long   _first_observed_frame =0; // 第一次被观测到的帧
    map<unsigned long, Vector3d> _obs;   // observations, first=frame ID, second=(pixel coordinate, depth), depth by default is 1 
    
    bool            _bad =false;        // bad 说的是这个点是不是很少被看到
    bool            _converged =false;  // 深度值是否收敛？
    
    // ORB feature 
    cv::Mat         _descriptor;        // 描述子
    cv::KeyPoint    _keypoint;          // 关键点
    
public:
    
    // 获得某个观测的相机坐标值
    Vector3d GetObservedPt( const unsigned long& keyframe_id );
    
    // for debug use
    void PrintInfo() {
        LOG(INFO) << "map point " << _id << "\nworld pos = " << _pos_world.transpose()<<endl;
        LOG(INFO) << "first observed from " << _first_observed_frame << endl;
        LOG(INFO) << "observations: "<<endl;
        for ( auto obs: _obs ) {
            LOG(INFO) << "from frame "<<obs.first << ", pixel pos = " << obs.second.transpose() << endl;
        }
    }
}; 
}

#endif 
