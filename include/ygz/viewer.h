#ifndef YGZ_VIEWER_H_
#define YGZ_VIEWER_H_

#include <pangolin/pangolin.h>

#include "ygz/memory.h"
#include "ygz/local_mapping.h"

namespace ygz
{


// 使用胖果林画出相机的 pose
class Viewer
{
    friend class Memory;
    friend class LocalMapping;

public:
    Viewer();

    // 画
    void Draw() {
        glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
        _dcam.Activate ( _scam );
        glClearColor ( 1.0f, 1.0f, 1.0f, 1.0f );
        // draw the key-frames and map points in memory

        for ( auto frame: Memory::_frames ) {
            DrawPose ( frame.second->_T_c_w, 1.0, 0, 0 );
        }

        DrawPose ( _curr_pose, 0.0, 0.0, 1.0 );

        // temp pose
        for ( SE3& pose: _poses_tmp ) {
            DrawPose ( pose, false );
        }

        // draw the map points
        for ( auto point : Memory::_points ) {
            if ( point.second->_converged ) {
                glPointSize ( 3 );
                glBegin ( GL_POINTS );
                glColor3d ( 0.1,0.6,0.1 );
                glVertex3d ( point.second->_pos_world[0], point.second->_pos_world[1], point.second->_pos_world[2] );
                glEnd();
            } else {
                glPointSize ( 3 );
                glBegin ( GL_POINTS );
                glColor3d ( 0.9,0.1,0.1 );
                glVertex3d ( point.second->_pos_world[0], point.second->_pos_world[1], point.second->_pos_world[2] );
                glEnd();
            }
        }

        // 画当前帧对地图点的观测
        if ( _current ) {
            Vector3d p_cam = _current->_T_c_w.inverse().translation();
            for ( unsigned long& id: _current->_map_point ) {
                MapPoint::Ptr mp = Memory::GetMapPoint ( id );
                Vector3d p = mp->_pos_world;
                glBegin ( GL_LINES );
                glColor3f ( 0,1.0,0 );
                glVertex3f ( p_cam[0], p_cam[1], p_cam[2] );
                glVertex3f ( p[0], p[1], p[2] );
                glEnd();

            }
        }

        pangolin::FinishFrame();
        usleep ( 1000 );
    }

    // 增加一些位姿
    void AddTempPose ( const SE3& new_pose ) {
        _poses_tmp.push_back ( new_pose );
        if ( _poses_tmp.size() > 100 ) {
            _poses_tmp.pop_front();
        }
    }

    inline void SetCurrPose ( const SE3& curr_pose ) {
        _curr_pose = curr_pose;
    }

    inline void SetCurrFrame ( const Frame::Ptr& current ) {
        _current = current;
        LOG ( INFO ) << _current->_id<<endl;
    }

private:
    // 画一个相机
    void DrawPose ( const SE3& T_c_w, float r=0, float g=0, float b=0 );

    pangolin::View _dcam;
    pangolin::OpenGlRenderState _scam;

    deque<SE3> _poses_tmp;     // 缓存一些中间的 pose ，如果太多就删掉一些早先的
    SE3 _curr_pose;

    LocalMapping* _local_mapping =nullptr;       // 想要画出LocalMapping中的位姿和地图点

public:
    Frame::Ptr _current;
};

}


#endif
