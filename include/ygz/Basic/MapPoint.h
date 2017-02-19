#ifndef YGZ_MAP_POINT_H_
#define YGZ_MAP_POINT_H_

#include "ygz/Basic/Common.h"

namespace ygz {
    
struct Feature;

// Map Point 代表一个地图点
// 它记录了自己在世界坐标系的位置，以及过去被观测到的情况
// map points 由Memory管理，要向 Ｍemory 注册以获得合法的id
// map point 由若干个Feature三角化而来，所以存储这些Feature的指针
// 另外 map point 也有一些统计量，用于判断它的好坏
// 如果map point有了好的深度值，就可以用于直接法位姿估计以及image alignment，否则，我们需要在极线上进行搜索，才能得到正确的深度
    
struct MapPoint {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    MapPoint()  {}
    
    // 获得匹配到/看到的比例
    inline float GetFoundRatio() const {
        return (float) _cnt_found/_cnt_visible;
    }
        
    // 计算有代表性的描述子
    // void ComputeDistinctiveDesc(); 
    
public:
    unsigned long   _id =0;                             // 全局id
    Vector3d        _pos_world =Vector3d(0,0,0);        // 世界坐标
    
    // 所有的 observations
    map<unsigned long, Feature*>  _obs;
    
    bool            _bad=false;         // bad flag
    
    Mat             _distinctive_desc;  // 描述子，代表这个地图点
    
    unsigned long   _first_seen =0;     // 第一次看到的帧(关键帧id)
    unsigned long   _last_seen =0;      // 最后一次看到的帧(普通帧id)
    int             _cnt_visible =0;    // 被看到的次数
    int             _cnt_found =0;      // 被匹配到的次数
    bool            _track_in_view =false; // 是否在视野中 
}; 
}

#endif // YGZ_MAP_POINT_H_
