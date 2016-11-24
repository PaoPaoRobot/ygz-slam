#ifndef VISUAL_ODOMETRY_H_
#define VISUAL_ODOMETRY_H_

#include "ygz/frame.h"
#include "ygz/tracker.h"
#include "ygz/initializer.h"

namespace ygz {

class System;
    
class VisualOdometry {
public:
    enum Status { 
        VO_NOT_READY,
        VO_INITING,
        VO_GOOD,
        VO_LOST,
        VO_ERROR,
    };
    
    VisualOdometry( System* system ) : _tracker(new Tracker)
    { 
        _system = system; 
    }
    
    void addFrame( const Frame::Ptr& frame );
    
    void plotFrame() const {
        _tracker->PlotTrackedPoints();
    }
    
    void setKeyframe( Frame::Ptr frame ); 
    
protected:
    void MonocularInitialization();
    
protected:
    Status _status =VO_NOT_READY;       // current status 
    Status _last_status;                // last status 
    
    Frame::Ptr  _curr_frame=nullptr;    // current 
    Frame::Ptr  _ref_frame=nullptr;     // reference 

    System* _system;                    // point to full system 
    unique_ptr<Tracker>  _tracker;      // tracker, most LK flow
    Initializer _init;                  // initializer  
};
}

#endif
