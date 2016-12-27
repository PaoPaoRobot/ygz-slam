#ifndef LOCAL_MAPPING_H
#define LOCAL_MAPPING_H

#include "ygz/common_include.h"
#include "ygz/utils.h"
#include "ygz/frame.h"
#include "ygz/map_point.h"

using namespace ygz::utils;

namespace ygz {

/****************************************************************
 * Local Mapping 是局部地图，指与当前帧相邻的几个附近的关键帧
 * 以及在当前帧视野范围内的地图点
 * 请注意可能有一些候选的地图点，它们的三维位置尚未确定，但二维位置由某个关键帧给出
 ****************************************************************/

class LocalMapping {
    
//class Frame;
// class MapPoint;
    
    struct Options {
        int min_track_localmap_inliers = 50;
    } _options;
    
public:
    LocalMapping();
    // 向局部地图增加一个关键帧，同时向局部地图中添加此关键帧关联的地图点
    void AddKeyFrame( Frame* keyframe );
    
    // 寻找地图与当前帧之间的匹配，当前帧需要有位姿的粗略估计，如果匹配顺利，进一步优化当前帧的 pose
    bool TrackLocalMap( Frame* current );
    
    // 新增一个路标点
    void AddMapPoint( const unsigned long& map_point_id );
    
    // 更新局部关键帧
    void UpdateLocalKeyframes( Frame* current ); 
    
    // 更新局部地图点
    void UpdateLocalMapPoints( Frame* current );
    
    
private:
    // 测试某个点是否可以和当前帧匹配上
    bool TestDirectMatch( Frame* current, const MatchPointCandidate& candidate, Vector2d& px_curr );
    
    // Local Bundle Adjustment 
    // 这里只修改current的pose，以及与current关联的路标点，不会修改其他的路标点和帧的位姿
    // 用于刚加入新的普通帧时的优化
    void LocalBA( Frame* current ); 
    
    // 局部关键帧和地图点
    set<Frame*> _local_keyframes;        
    set<MapPoint*> _local_map_points;   
    
    // 匹配局部地图用的 patch
    uchar _patch[WarpPatchSize*WarpPatchSize];
    // 带边界的，左右各1个像素
    uchar _patch_with_border[(WarpPatchSize+2)*(WarpPatchSize+2)];
    
    // parameters 
    int _num_local_keyframes =3; // 相邻关键帧数量
    int _num_local_map_points =500;
    
    int _image_width=640, _image_height=480;
    int _cell_size;
    int _grid_rows=0, _grid_cols=0;
    int _pyramid_level=0;
    
    vector<vector<MatchPointCandidate>> _grid;  // 候选点网格,每个格点有一串的候选点
};
}

#endif
