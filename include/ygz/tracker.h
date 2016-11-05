#ifndef TRACKER_H_
#define TRACKER_H_

#include "ygz/frame.h"

namespace ygz {
    
class Tracker {
public:
    enum TrackerStatusType {
        NOT_READY,
        GOOD,
        LOST
    };
    
    Tracker();
    ~Tracker(); 
    
    void SetReference( Frame::Ptr ref );
    void Track( Frame::Ptr curr );   
    
    TrackerStatusType Status() const { return _status; }
    
    void DetectFeatures( 
        Frame::Ptr frame,
        vector<Vector2d>& pts
    );
    
protected:
    Frame::Ptr _ref =nullptr;            // reference 
    Frame::Ptr _curr =nullptr;           // current  
    vector<Vector2d> _px_ref;           // pixels in ref 
    vector<Vector2d> _px_curr;           // pixels in ref 
    TrackerStatusType _status =NOT_READY;
    
    // parameters 
    int _min_features_initializing; 
};

}

#endif