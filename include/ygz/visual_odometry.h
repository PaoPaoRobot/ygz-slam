#ifndef VISUAL_ODOMETRY_H_
#define VISUAL_ODOMETRY_H_

#include "ygz/frame.h"
#include "ygz/optimizer.h"

namespace ygz {

class System;
class Memory;
class Tracker;
class Initializer;
class FeatureDetector;
class LocalMapping;
class ORBExtractor;
class ORBMatcher;
    
class VisualOdometry {
    friend class Memory;
public:
    enum Status { 
        VO_NOT_READY,
        VO_INITING,
        VO_GOOD,
        VO_LOST,
        VO_ERROR,
    };
    
    struct Option {
        double init_reproj_error_th =4; // 初始化时判断内点的最大重投影误差
        double max_sparse_align_motion=0.5; // sparse alignment最大运动，避免出错
    } _options;
    
    VisualOdometry( System* system );
    virtual ~VisualOdometry();
    
    // 新增一个帧，如果该帧可以顺利跟踪，返回真
    bool AddFrame( Frame* frame );
    
protected:
    // set the input frame as a key-frame 
    // 第二个参数表示是否在初始化中使用
    void SetKeyframe( Frame* frame, bool initializing = false ); 
    
    // 跟踪最近的帧
    bool TrackRefFrame();
    
    // 跟踪局部地图
    bool TrackLocalMap();
    // 单目初始化
    void MonocularInitialization();
    
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
    
private:
    Status _status =VO_NOT_READY;       // current status 
    Status _last_status;                        // last status 
    
    Frame*  _curr_frame=nullptr;                // current 
    Frame*  _ref_frame=nullptr;                 // reference,通常是上一个帧，和当前帧最相似 

    System* _system=nullptr;                    // point to full system 
    Tracker*  _tracker=nullptr;                 // tracker, most LK flow
    Initializer* _init =nullptr;                 // initializer  
    
    // sparse image alignment for each pyramid level 
    vector<opti::SparseImgAlign, Eigen::aligned_allocator<opti::SparseImgAlign>> _align;
    
    LocalMapping* _local_mapping=nullptr;       // 局部地图
    
    opti::DepthFilter* _depth_filter =nullptr;  // depth filter 
    
    FeatureDetector* _detector=nullptr;         // feature detection 
    
    ORBExtractor* _orb_extractor =nullptr;      // ORB 提取
    ORBMatcher* _orb_matcher =nullptr;          // ORB 比较
    
    SE3 _TCR_estimated;                 // estimated transform from ref to current 
    
    // 上一个关键帧，这在判断是否产生新关键帧时有用
    Frame* _last_key_frame = nullptr;
    
    // params 
    // 关键帧选择策略中的最小旋转和最小平移量
    double _min_keyframe_rot=0;
    double _min_keyframe_trans=0;
    int _min_keyframe_features=0;
    
    int _processed_frames=0;      // 已经处理过的帧，用于判断是否插入关键帧
    
};
}

#endif
