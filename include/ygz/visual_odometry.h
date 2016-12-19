#ifndef VISUAL_ODOMETRY_H_
#define VISUAL_ODOMETRY_H_

#include "ygz/frame.h"
#include "ygz/tracker.h"
#include "ygz/initializer.h"
#include "ygz/optimizer.h"
#include "ygz/ORB/ORBextractor.h"
#include "ygz/local_mapping.h"

namespace ygz {

class System;
class Memory;
    
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
        double init_reproj_error_th =5; // 初始化时判断内点的最大重投影误差
    } _options;
    
    VisualOdometry( System* system );
    virtual ~VisualOdometry();
    
    // 新增一个帧，如果该帧可以顺利跟踪，返回真
    bool AddFrame( const Frame::Ptr& frame );
    
    // 画出跟踪的地图点
    void PlotFrame() const {
        _tracker->PlotTrackedPoints();
    }
    
protected:
    // set the input frame as a key-frame 
    // 第二个参数表示是否在初始化中使用
    void SetKeyframe( Frame::Ptr& frame, bool initializing = false ); 
    
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
    
private:
    Status _status =VO_NOT_READY;       // current status 
    Status _last_status;                // last status 
    
    Frame::Ptr  _curr_frame=nullptr;    // current 
    Frame::Ptr  _ref_frame=nullptr;     // reference 

    System* _system;                    // point to full system 
    shared_ptr<Tracker>  _tracker;      // tracker, most LK flow
    Initializer _init;                  // initializer  
    vector<opti::SparseImgAlign> _align;// sparse image alignment for each pyramid level 
    LocalMapping        _local_mapping; // 局部地图
    opti::DepthFilter*  _depth_filter =nullptr;  // depth filter 
    shared_ptr<FeatureDetector> _detector=nullptr;
    
    SE3 _TCR_estimated;                 // estimated transform from ref to current 
    
    // 上一个关键帧，这在判断是否产生新关键帧时有用
    Frame::Ptr _last_key_frame = nullptr;
    
    // 关键帧选择策略中的最小旋转和最小平移量
    double _min_keyframe_rot;
    double _min_keyframe_trans;
    int _min_keyframe_features;
    
    int _image_width=640, _image_height=480;    // 图像长宽，用以计算网格
    int _cell_size;                             // 网格大小
    int _grid_rows=0, _grid_cols=0;             // 网格矩阵的行和列
    double _detection_threshold =20.0;          // 特征响应阈值
};
}

#endif
