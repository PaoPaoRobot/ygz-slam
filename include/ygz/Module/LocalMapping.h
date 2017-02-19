#ifndef YGZ_LOCAL_MAPPING_H_
#define YGZ_LOCAL_MAPPING_H_

#include "ygz/Basic.h"
#include "ygz/Algorithm.h"


namespace ygz 
{
    
class LocalMapping 
{
public:
   
    struct Options 
    {
        int _num_local_keyframes =3; // 相邻关键帧数量
        int _num_local_map_points =500;
    };
    
    LocalMapping(); 
    
    // 向局部地图增加一个关键帧，同时向局部地图中添加此关键帧关联的地图点
    void AddKeyFrame( Frame* keyframe );
    
    // 寻找地图与当前帧之间的匹配，当前帧需要有位姿的粗略估计，如果匹配顺利，进一步优化当前帧的 pose
    bool TrackLocalMap( Frame* current );
    
    // 新增一个路标点
    void AddMapPoint( MapPoint* mp );
    
    // 更新局部关键帧
    void UpdateLocalKeyframes( Frame* current ); 
    
    // 更新局部地图点
    void UpdateLocalMapPoints( Frame* current );
    
    // local mapping 优化线程
    void Run();
    
    // 处理新的关键帧
    void ProcessNewKeyFrame(); 
    
    // 删除追踪不好的地图点
    void MapPointCulling(); 
    
    // 新建一些地图点
    void CreateNewMapPoints(); 
    
    // 在当前关键帧二级相邻的帧中搜索匹配点
    void SearchInNeighbors();
    
    // 修正 keyframe 
    void KeyFrameCulling(); 
    
private:
    // Local Bundle Adjustment 
    void LocalBA( Frame* current ); 
    
    // Optimize current frame and the associated features 
    void OptimizeCurrent( Frame* current );
    
    // Find candidates in local map points 
    map<Feature*, Vector2d> FindCandidates( Frame* current );
    
    // Project map points into current 
    void ProjectMapPoints( Frame* current, map<Feature*, Vector2d>& candidates );
    
private:
    // Data 
    // 局部关键帧和地图点，用于tracking
    std::set<Frame*> _local_keyframes;        
    std::set<MapPoint*> _local_map_points;   
    
    // 新增的关键帧和地图点
    std::list<Frame*> _new_keyframes;
    std::list<MapPoint*> _new_mappoints;
    
    // 当前正在处理的关键帧
    Frame* _current_kf;
    
    // 新增的一些地图点
    std::list<MapPoint*> _recent_mappoints;
    
    // 匹配算法
    Matcher* _matcher =nullptr;
};
    
}

#endif