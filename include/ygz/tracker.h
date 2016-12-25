#ifndef TRACKER_H_
#define TRACKER_H_

#include "ygz/frame.h"

namespace ygz {
    
// Tracker
// 用来跟踪特征点
// Tracker 一般只计算两个帧的匹配关系，它默认使用光流跟踪
// 如果使用特征点的活，也可以匹配特征点的描述 
    
class FeatureDetector;
    
class Tracker {
public:
    // Tracker 的状态，只有一个帧时为 NOT READY，顺利跟踪则为 GOOD，否则为 LOST
    enum TrackerStatusType {
        TRACK_NOT_READY,
        TRACK_GOOD,
        TRACK_LOST
    };
    
public:
    Tracker( FeatureDetector* detector );
    
    // set the reference to track 
    void SetReference( Frame* ref );
    
    // track the frame, call it after setting the reference
    void Track( Frame* curr );   
    
    // compute mean disparity, used in initilization 
    float MeanDisparity() const ;
    
    void GetTrackedPointsNormalPlane (
        vector<Vector2d>& pt1, 
        vector<Vector2d>& pt2 
    ) const ;
    
    void GetTrackedPixel (
        vector<Vector2d>& px1, 
        vector<Vector2d>& px2 
    ) const ;
    
    // draw the tracked points 
    void PlotTrackedPoints() const; 
    
    // accessors 
    TrackerStatusType Status() const { return _status; }
    
    list<cv::Point2f> GetPxCurr() const { return _px_curr; }
    
private:
    void TrackKLT( );
    
private:
    Frame* _ref =nullptr;            // reference 
    Frame* _curr =nullptr;           // current  
    
    list<cv::Point2f> _px_curr;           // pixels in curr, lost features will be deleted from reference 
    
    TrackerStatusType _status =TRACK_NOT_READY;
    FeatureDetector* _detector =nullptr; 
    
    // parameters 
    int _min_features_initializing; 
};

}

#endif