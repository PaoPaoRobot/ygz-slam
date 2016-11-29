#ifndef MEMORY_H_
#define MEMORY_H_

#include "ygz/frame.h"
#include "ygz/map_point.h"

namespace ygz {
    
class Viewer;
    
class Memory {
    friend Viewer;
public:
    ~Memory() { 
        if ( _mem != nullptr )
            _mem->clean();
    }
    
    static Frame::Ptr CreateNewFrame(); 
    
    static Frame::Ptr CreateNewFrame(
        const double& timestamp, 
        const SE3& T_c_w, 
        const bool is_keyframe, 
        const Mat& color, 
        const Mat& depth = Mat()
    ); 
    
    // register a frame into memory, assign an ID with it
    // if already registered, you can choose whether to overwrite the exist one 
    static Frame::Ptr RegisterFrame( Frame::Ptr& frame, bool overwrite=false );
    
    static MapPoint::Ptr CreateMapPoint(); 
    
    static inline Frame::Ptr GetFrame( const unsigned long& id ) {
        auto iter = _frames.find( id );
        if ( iter == _frames.end() )
            return nullptr;
        return iter->second;
    }
    
    static inline MapPoint::Ptr GetMapPoint( const unsigned long& id ) {
        auto iter = _points.find( id );
        if ( iter == _points.end() )
            return nullptr;
        return iter->second;
    }
    
    void optimizeMemory(); 
    void clean();
    
    
private:
    Memory() {}
    
protected:
    static unordered_map<unsigned long, Frame::Ptr> _frames; 
    static unordered_map<unsigned long, MapPoint::Ptr> _points;
    static unsigned long _id_frame, _id_points; 
    
    static shared_ptr<Memory> _mem;
};
    
}

#endif
