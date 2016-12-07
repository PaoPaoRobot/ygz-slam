#ifndef LOCAL_MAPPING_H
#define LOCAL_MAPPING_H

#include "ygz/common_include.h"
#include "ygz/frame.h"
#include "ygz/map_point.h"
#include "ygz/utils.h"

using namespace ygz::utils;

namespace ygz {

/****************************************************************
 * Local Mapping 是局部地图，指与当前帧相邻的几个附近的关键帧
 * 以及在当前帧视野范围内的地图点
 * 请注意可能有一些候选的地图点，它们的三维位置尚未确定，但二维位置由某个关键帧给出
 ****************************************************************/


class LocalMapping {
    
private:
    // 测试某个点是否可以和当前帧匹配上
    bool TestDirectMatch( Frame::Ptr current, const MatchPointCandidate& candidate, Vector2d& px_curr );
    
    // 相邻关键帧和地图点，以ID形式标出
    set<unsigned long> _local_keyframes;        // 
    set<unsigned long> _local_map_points;       // 
    
    // 匹配局部地图用的 patch
    uchar _patch[WarpPatchSize*WarpPatchSize];
    // 带边界的，左右各1个像素
    uchar _patch_with_border[(WarpPatchSize+2)*(WarpPatchSize+2)];
    
    // parameters 
    int _num_local_keyframes =3; // 相邻关键帧数量
    
    int _image_width=640, _image_height=480;
    int _cell_size;
    int _grid_rows=0, _grid_cols=0;
    int _pyramid_level=0;
    
    vector<vector<MatchPointCandidate>> _grid;  // 候选点网格,每个格点有一串的候选点
    
public:
    LocalMapping();
    // 向局部地图增加一个关键帧，同时向局部地图中添加此关键帧关联的地图点
    void AddKeyFrame( Frame::Ptr keyframe );
    
    // 寻找地图与当前帧之间的匹配，当前帧需要有位姿的粗略估计
    list<MatchPointCandidate> FindMatchedCandidate( Frame::Ptr current );

};
}

#endif
