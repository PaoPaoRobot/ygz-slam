#include "ygz/Basic/Memory.h"
#include "ygz/Basic/Frame.h"
#include "ygz/Basic/MapPoint.h"

namespace ygz {
    
void Memory::Clean()
{
    for ( auto iter = _frames.begin(); iter!=_frames.end(); iter++ ) {
        delete iter->second;
    }
    _mem->_frames.clear();
    
    for ( auto& point_pair: _mem->_points ) {
        delete point_pair.second;
    }
    _mem->_points.clear();
}

Frame* Memory::RegisterKeyFrame(Frame* frame, bool overwrite)
{
    // check if this frame has already registered 
    if ( _mem->_frames.find(frame->_keyframe_id) != _mem->_frames.end() ) {
        // already registered 
        if ( overwrite ) {
            _mem->_frames[ frame->_keyframe_id ] = frame; 
            return frame; 
        } else {
            frame->_keyframe_id = _mem->_id_frame; 
            _mem->_frames[ frame->_keyframe_id ] = frame; 
            _mem->_id_frame++;
            return frame; 
        }
    } else {
        // set its id 
        frame->_keyframe_id = _mem->_id_frame; 
        _mem->_frames[ frame->_keyframe_id ] = frame; 
        _mem->_id_frame++;
        return frame; 
    }
}


MapPoint* Memory::CreateMapPoint()
{
    MapPoint* pm = new MapPoint;
    pm->_id = _mem->_id_points;
    _mem->_id_points++;
    _mem->_points[ pm->_id ] = pm;
    return pm;
}

Frame* Memory::GetKeyFrame( const unsigned long& keyframe_id )
{
    auto iter = _mem->_frames.find( keyframe_id );
    if ( iter == _mem->_frames.end() ) {
        return nullptr;
    }
    return iter->second;
}

MapPoint* Memory::GetMapPoint( const unsigned long& id ) {
    auto iter = _mem->_points.find( id );
    if ( iter == _mem->_points.end() )
        return nullptr;
    return iter->second;
}

shared_ptr<Memory> Memory::_mem( new Memory() ) ;

}