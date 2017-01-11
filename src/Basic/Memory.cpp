#include "ygz/Basic/Memory.h"
#include "ygz/Basic/Frame.h"
#include "ygz/Basic/MapPoint.h"

namespace ygz {
    
void Memory::Clean()
{
    for ( auto& frame_pair: _frames ) {
        delete frame_pair.second;
    }
    _frames.clear();
    
    for ( auto& point_pair: _points ) {
        delete point_pair.second;
    }
    _points.clear();
}

Frame* Memory::RegisterKeyFrame(Frame* frame, bool overwrite)
{
    // check if this frame has already registered 
    if ( _frames.find(frame->_keyframe_id) != _frames.end() ) {
        // already registered 
        if ( overwrite ) {
            _frames[ frame->_keyframe_id ] = frame; 
            return frame; 
        } else {
            frame->_keyframe_id = _id_frame; 
            _frames[ frame->_keyframe_id ] = frame; 
            _id_frame++;
            return frame; 
        }
    } else {
        // set its id 
        frame->_keyframe_id = _id_frame; 
        _frames[ frame->_keyframe_id ] = frame; 
        _id_frame++;
        return frame; 
    }
}


MapPoint* Memory::CreateMapPoint()
{
    MapPoint* pm = new MapPoint;
    pm->_id = _id_points;
    _id_points++;
    _points[ pm->_id ] = pm;
    return pm;
}

Frame* Memory::GetKeyFrame( const unsigned long& keyframe_id )
{
    auto iter = _frames.find( keyframe_id );
    if ( iter == _frames.end() ) {
        return nullptr;
    }
    return iter->second;
}

MapPoint* Memory::GetMapPoint( const unsigned long& id ) {
    auto iter = _points.find( id );
    if ( iter == _points.end() )
        return nullptr;
    return iter->second;
}

shared_ptr<Memory> Memory::_mem(new Memory) ;
unsigned long Memory::_id_frame =0;
unsigned long Memory::_id_points =0; 

unordered_map<unsigned long, Frame*> Memory::_frames; 
unordered_map<unsigned long, MapPoint*> Memory::_points;

}