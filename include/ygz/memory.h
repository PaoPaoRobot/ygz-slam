#ifndef MEMORY_H_
#define MEMORY_H_

#include "ygz/frame.h"
#include "ygz/map_point.h"

namespace ygz {
    
class Memory {
public:
    ~Memory() { 
        if ( _mem != nullptr )
            _mem->clean();
    }
    
    inline void InitMemory() { 
        if ( _mem == nullptr )
            _mem = make_shared<Memory>( );
    }
        
    static Frame::Ptr CreateNewFrame(); 
    static MapPoint::Ptr CreateMapPoint(); 
    
    static inline Frame::Ptr Frame( const unsigned long& id ) {
        auto iter = _frames.find( id );
        if ( iter == _frames.end() )
            return nullptr;
        return *iter;
    }
    
    static inline MapPoint::Ptr MapPoint( const unsigned long& id ) {
        auto iter = _points.find( id );
        if ( iter == _points.end() )
            return nullptr;
        return *iter;
    }
    
    void optimizeMemory(); 
    void clean();
    
    
private:
    Memory() {}
    
protected:
    static unordered_map<unsigned long, Frame::Ptr> _frames; 
    static unordered_map<unsigned long, MapPoint::Ptr> _points;
    static unsigned long _id_frame=0, _id_points=0; 
    
    static shared_ptr<Memory> _mem;
};
    
}

#endif
