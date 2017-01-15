#ifndef YGZ_TRACKER_H_
#define YGZ_TRACKER_H_

#include "ygz/Basic.h"

namespace ygz 
{
    
// 基于光流法的 Tracker，需要给定参考帧和当前帧，Tracker会追踪参考帧中的特征点
// 当参考帧和当前帧有明显差异时，Tracker没法保证跟踪结果是正确的，需要算法进一步验证，例如计算描述的距离
class Tracker
{
public:
    // Tracker 的状态，只有一个帧时为 NOT READY，顺利跟踪则为 GOOD，否则为 LOST
    enum TrackerStatusType {
        TRACK_NOT_READY,
        TRACK_GOOD,
        TRACK_LOST
    };

    struct Option 
    {
        int _min_feature_tracking =50; // 最小追踪特征数量，小于此数目时认为丢失
        // KLT 参数
        double klt_win_size = 30.0;
        int klt_max_iter = 50;
        double klt_eps = 0.001;
    } _option;
    
    
    Tracker();
    
    // set the reference to track 
    void SetReference( Frame* ref );
    
    // track the frame, call it after setting the reference
    void Track( Frame* curr );   
    
    // compute mean disparity, used in initilization 
    float MeanDisparity() const ;
    
    // 获取 Tracker 得到的特征
    void GetTrackedPixel (
        vector<Feature*>& feature1, 
        vector<Vector2d>& pixels2 
    ) const ;
    
    // draw the tracked points 
    void PlotTrackedPoints() const; 
    
    // accessors 
    TrackerStatusType Status() const { return _status; }

private:
    
    // L-K Optical flow tracking, use OpenCV's function 
    void TrackKLT();
    
    Frame* _ref  =nullptr;
    Frame* _curr =nullptr;
    
    list<Feature*> _tracked_features;   // 参考帧中被顺利追踪的特征
    vector<cv::Point2f> _px_curr;         // 参考帧中特征点在当前帧上的位置
    TrackerStatusType _status =TrackerStatusType::TRACK_NOT_READY;
    
};
    
}


#endif // YGZ_TRACKER_H_