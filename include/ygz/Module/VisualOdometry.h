#ifndef YGZ_VISUAL_ODOMETRY_H_
#define YGZ_VISUAL_ODOMETRY_H_

#include "ygz/Basic.h"
#include "ygz/Algorithm.h"

namespace ygz 
{

class System;     

/**
 * @brief VisualOdometry 视觉里程计模块
 */
class VisualOdometry
{
public:
    // 里程计的状态，单目的里程计需要初始化
    enum Status 
    { 
        VO_NOT_READY,
        VO_INITING,
        VO_GOOD,
        VO_LOST,
        VO_ERROR,
    };
    
    
    // 配置参数
    struct Option 
    {
        double init_reproj_error_th =4; // 初始化时判断内点的最大重投影误差
        double max_sparse_align_motion=0.5; // sparse alignment最大运动，避免出错
        
        // 关键帧选择策略中的最小旋转和最小平移量
        double _min_keyframe_rot=0;
        double _min_keyframe_trans=0;
        int _min_keyframe_features=0;
        
        float _min_init_disparity =40; // 初始化时最小平均视差，大于它才会调用初始化
        int _min_init_features =40;   // 最小初始化时误差
        int _processed_frames=0;      // 已经处理过的帧，用于判断是否插入关键帧
    } _options;
    
    
public:
    VisualOdometry( System* system =nullptr );
    virtual ~VisualOdometry();
    
    // 新增一个帧，如果该帧可以顺利跟踪，返回真
    /**
     * @brief Add a new frame into visual odometry 
     * @param[in] frame the input frame 
     * @returns true if tracked correctly
     */
    bool AddFrame( Frame* frame );
    
private:
    // 设置新的关键帧。注意我们仅在关键帧中提取新特征点
    void SetKeyframe( Frame* frame ); 
    
    // 跟踪最近的帧
    bool TrackRefFrame();
    
    // 跟踪局部地图
    bool TrackLocalMap();
    
    // 根据优化后的位姿和地图点，对初始化的地图点计算投影位置
    void ReprojectMapPointsInInitializaion(); 
    
    // 检查当前帧是否可以成为新的关键帧
    bool NeedNewKeyFrame(); 
    
    // 重置当前帧中的观测信息
    void ResetCurrentObservation(); 
    
    // 初始化中，根据描述子检测两个帧跟踪的点是否成立
    // 如果成立的话，将在地图中新增一些地图点
    bool CheckInitializationByDescriptors( );  
    
    // 添加关键帧时，通过描述来计算某个关键帧的观测量是否正确
    bool CheckObservationByDescriptors(); 
    
    /// 初始化相关函数 /// 
    // 单目初始化
    bool MonocularInitialization();
    // 初始化之后处理地图点
    void CreateMapPointsAfterMonocularInitialization(
        vector<Feature*>& features_ref, 
        vector<Vector2d>& pixels_curr,
        vector<Vector3d>& pts_triangulated,
        vector<bool>& inliers
    ); 
    
    
private:
    Status _status =VO_NOT_READY;               // current status 
    Status _last_status=VO_NOT_READY;           // last status 
    
    Frame*  _curr_frame=nullptr;         
    Frame*  _ref_frame=nullptr;          

    Tracker*            _tracker=nullptr; 
    Initializer*        _init =nullptr;  
    ygz::Matcher*       _matcher=nullptr;
    FeatureDetector*    _detector=nullptr;         // feature detection 
    System*             _system=nullptr;
    
    SE3 _TCR_estimated;   // estimated transform from ref to current 
    
    // 上一个关键帧，这在判断是否产生新关键帧时有用
    Frame* _last_key_frame = nullptr;
    
};
    
}


#endif