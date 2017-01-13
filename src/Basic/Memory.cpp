#include "ygz/Basic/Memory.h"
#include "ygz/Basic/Frame.h"
#include "ygz/Basic/MapPoint.h"

namespace ygz {
    
void Memory::Clean()
{
    LOG(INFO)<<"Cleaning memory"<<endl;
    LOG(INFO) << "frames: "<<_frames.size()<<endl;
    // for ( auto iter = _frames.begin(); iter!=_frames.end(); iter++ ) {
        // LOG(INFO) << "delete frame "<<iter->first<<endl;
        // delete iter->second;
    // }
    // LOG(INFO)<<"clear frames"<<endl;
    // _frames.clear();
    
    LOG(INFO) << "map points: "<<_points.size()<<endl;
    /*
    for ( auto& point_pair: _points ) {
        LOG(INFO)<<"delete map point "<<point_pair.first<<endl;
        delete point_pair.second;
    }
    _points.clear();
    */
    _frames.clear();
    _points.clear();
    LOG(INFO)<<"Cleaning memory returns"<<endl;
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

void Memory::Init()
{
    _mem = make_shared<Memory>();
}


shared_ptr<Memory> Memory::_mem( nullptr ) ;
unsigned long Memory::_id_frame =0;
unsigned long Memory::_id_points =0; 

map<unsigned long, Frame*> Memory::_frames; 
map<unsigned long, MapPoint*> Memory::_points;

}