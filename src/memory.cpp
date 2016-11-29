#include "ygz/memory.h"

namespace ygz {
    
void Memory::clean()
{
    _frames.clear();
    _points.clear();
}

Frame::Ptr Memory::CreateNewFrame()
{
    Frame::Ptr pf ( new Frame );
    pf->_id = _id_frame; 
    _frames[pf->_id] = pf;
    _id_frame++;
    return pf;
}

Frame::Ptr Memory::CreateNewFrame(
    const double& timestamp, const SE3& T_c_w, const bool is_keyframe, 
    const Mat& color, const Mat& depth )
{
    Frame::Ptr pf ( new Frame(
        timestamp, T_c_w, is_keyframe, color, depth
    ) );
    pf->_id = _id_frame; 
    pf->InitFrame(); 
    _frames[pf->_id] = pf;
    _id_frame++;
    return pf;
}

Frame::Ptr Memory::RegisterFrame(Frame::Ptr& frame, bool overwrite)
{
    // check if this frame has already registered 
    if ( _frames.find(frame->_id) != _frames.end() ) {
        // already registered 
        if ( overwrite ) {
            _frames[ frame->_id ] = frame; 
            return frame; 
        } else {
            frame->_id = _id_frame; 
            _frames[ frame->_id ] = frame; 
            _id_frame++;
            return frame; 
        }
    } else {
        // set its id 
        frame->_id = _id_frame; 
        _frames[ frame->_id ] = frame; 
        _id_frame++;
        return frame; 
    }
}


MapPoint::Ptr Memory::CreateMapPoint()
{
    MapPoint::Ptr pm ( new MapPoint );
    pm->_id = _id_points;
    _id_points++;
    _points[ pm->_id ] = pm;
    return pm;
}

shared_ptr<Memory> Memory::_mem(new Memory) ;
unsigned long Memory::_id_frame =0;
unsigned long Memory::_id_points =0; 

unordered_map<unsigned long, Frame::Ptr> Memory::_frames; 
unordered_map<unsigned long, MapPoint::Ptr> Memory::_points;

}