#ifndef VISUAL_ODOMETRY_H_
#define VISUAL_ODOMETRY_H_

#include "ygz/frame.h"

namespace ygz {

class Frame;
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
    
    VisualOdometry( System* system ) { 
        _system = system; 
    }
    
    void addFrame( const Frame::Ptr& frame );
    
protected:
    void MonocularInitialization();
    
protected:
    Status _status =VO_NOT_READY; 
    Status _last_status;
    
    Frame::Ptr  _curr_frame=nullptr;    // current 
    Frame::Ptr  _ref_frame=nullptr;     // reference 

    System* _system; 
};
}

#endif
