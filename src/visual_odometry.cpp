#include "ygz/visual_odometry.h"


namespace ygz { 
    
void VisualOdometry::addFrame(const Frame::Ptr& frame)
{
    if ( _status == VO_NOT_READY ) {
        _status = VO_INITING;
    }
    
    _last_status = _status; 
    if ( _status == VO_INITING ) {
        // TODO initialization 
        if ( _status != VO_GOOD )
            return; 
    } else {
        // initialized, do tracking 
        if ( _status == VO_GOOD ) {
            // track ref or velocity 
        } else {
            // not good, relocalize 
            // TODO: relocalization 
        }
    }
    
    // store the result or other things 
}

}
