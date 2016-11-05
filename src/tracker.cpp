#include "ygz/tracker.h"

namespace ygz {
    
Tracker::Tracker( )
{
    // read the params from config 
    _min_features_initializing = Config::get<int>("init.min_features");
}

void Tracker::SetReference(Frame::Ptr ref)
{
    DetectFeatures( ref, _px_ref );
    if ( _px_ref.size() < _min_features_initializing ) {
        LOG(WARNING) << "Init frame has few features, try moving in more textured environment. " << endl;
        _status = NOT_READY;
        return; 
    }
    _ref = ref; 
    _px_curr = _px_ref;
}

void Tracker::Track(Frame::Ptr curr)
{

}


void Tracker::DetectFeatures(
    Frame::Ptr frame, 
    vector< Vector2d >& pts )
{

}


    
}