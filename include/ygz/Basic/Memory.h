#ifndef YGZ_MEMORY_H_
#define YGZ_MEMORY_H_

#include "ygz/Basic/Common.h"

namespace ygz {
    
class Viewer;
class VisualOdometry;
struct Frame;
struct MapPoint;
    
// 内存管理类
// 它存储了所有的关键帧和关键帧带着的地图点，也只有这些帧的 keyframe id是合法的，可以从memory中按照id来访问
// 单件，不允许复制
class Memory {
    friend Viewer;
    friend VisualOdometry;
    
public:
    Memory( const Memory& ) =delete;
    Memory& operator = ( const Memory& ) =delete;
    
    ~Memory() { 
        if ( _mem != nullptr )
            _mem->Clean();
        _mem = nullptr;
    }
    
    static void Init() ;
    
    // register a frame into memory, allocate an ID to it
    // if already registered, you can choose whether to overwrite the exist one 
    static Frame* RegisterKeyFrame( Frame* frame, bool overwrite=false );
    
    static MapPoint* CreateMapPoint(); 
    
    static Frame* GetKeyFrame( const unsigned long& keyframe_id );
    
    static inline int GetNumberFrames() { return _frames.size(); }
    
    static MapPoint* GetMapPoint( const unsigned long& id );
    
    static inline map<unsigned long, MapPoint*> & GetAllPoints() {
        return _points;
    }
    
    void Clean();
    
    static void PrintInfo() 
    {
        LOG(INFO)<<"Frames: "<<_frames.size()<<endl;
        for ( auto& frame_pair: _frames )
            LOG(INFO) << frame_pair.first<<endl;
        LOG(INFO) << "Map Points: "<<_points.size()<<endl;
        for ( auto& mp_pair: _points )
            LOG(INFO) << mp_pair.first << endl;
    }
    
private:
    Memory() {}
    
private:
    static map<unsigned long, Frame*>         _frames; 
    static map<unsigned long, MapPoint*>      _points;
    static unsigned long _id_frame, _id_points; 
    static shared_ptr<Memory> _mem;
};
    
}

#endif
