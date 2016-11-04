#ifndef VISUAL_ODOMETRY_H_
#define VISUAL_ODOMETRY_H_

#include "ygz/frame.h"

namespace ygz {

class Frame;
    
class VisualOdometry {
public:
    enum Status { 
        VO_NOT_READY,
        VO_INITING,
        VO_GOOD,
        VO_LOST,
        VO_ERROR,
    };
    
    void addFrame( const Frame::Ptr& frame );
    
protected:
    Status _status =VO_NOT_READY; 
    Status _last_status;
    
    Frame::Ptr  _curr_frame=nullptr;    // current 
    Frame::Ptr  _ref_frame=nullptr;     // reference 

};
}

#endif
