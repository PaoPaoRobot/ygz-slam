#ifndef TRACKER_H_
#define TRACKER_H_

#include "ygz/frame.h"
#include "ygz/feature_detector.h"

namespace ygz {
    

class Tracker {
public:
    enum TrackerStatusType {
        TRACK_NOT_READY,
        TRACK_GOOD,
        TRACK_LOST
    };
    
    Tracker();
    ~Tracker(); 
    
    void SetReference( Frame::Ptr ref );
    void Track( Frame::Ptr curr );   
    
    TrackerStatusType Status() const { return _status; }
    
protected:
    void TrackKLT( );
    
protected:
    Frame::Ptr _ref =nullptr;            // reference 
    Frame::Ptr _curr =nullptr;           // current  
    
    vector<cv::Point2f> _px_ref;           // pixels in ref 
    vector<cv::Point2f> _px_curr;           // pixels in curr 
    TrackerStatusType _status =TRACK_NOT_READY;
    
    shared_ptr<FeatureDetector> _detector; 
    
    // parameters 
    int _min_features_initializing; 
};

}

#endif