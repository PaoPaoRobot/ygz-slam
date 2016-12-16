#ifndef YGZ_VIEWER_H_
#define YGZ_VIEWER_H_

#include <pangolin/pangolin.h>

#include "ygz/memory.h"
#include "ygz/local_mapping.h"

namespace ygz {
    
    
// 使用胖果林画出相机的 pose 
class Viewer {
friend class Memory; 
friend class LocalMapping;

public:
    Viewer(); 
    
    // 画
    void Draw(); 
    
    // 增加一些位姿
    void AddTempPose( const SE3& new_pose ) {
        _poses_tmp.push_back( new_pose );
        if ( _poses_tmp.size() > 100 )
            _poses_tmp.pop_front();
    }
    
    inline void SetCurrPose( const SE3& curr_pose ) {
        _curr_pose = curr_pose;
    }
    
    inline void SetCurrFrame( const Frame::Ptr current ) {
        _current = current;
    }
    
private:
    // 画一个相机
    void DrawPose( const SE3& T_c_w, float r=0, float g=0, float b=0 );
    
    pangolin::View _dcam; 
    pangolin::OpenGlRenderState _scam;
    
    deque<SE3> _poses_tmp;     // 缓存一些中间的 pose ，如果太多就删掉一些早先的
    SE3 _curr_pose;
    
    LocalMapping* _local_mapping =nullptr;       // 想要画出LocalMapping中的位姿和地图点
    
    Frame::Ptr _current =nullptr; 
};
    
}


#endif 