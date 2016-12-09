#include "ygz/viewer.h"
#include "ygz/memory.h"

namespace ygz
{

Viewer::Viewer()
{
    pangolin::CreateWindowAndBind ( "YGZ-SLAM: GUI", 1024, 768 );

    // 3D Mouse handler requires depth testing to be enabled
    glEnable ( GL_DEPTH_TEST );

    // Issue specific OpenGl we might need
    glEnable ( GL_BLEND );
    glBlendFunc ( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );

    // camera object
    _scam = pangolin::OpenGlRenderState (
                pangolin::ProjectionMatrix ( 640,480,300,300,320,240,0.1,1000 ),
                pangolin::ModelViewLookAt ( 0,-0.7, -1.8, 0,0,0,0.0,-1.0, 0.0 )
            );

    // Add named OpenGL viewport to window and provide 3D Handler
    _dcam = pangolin::CreateDisplay()
            .SetBounds ( 0.0, 1.0, pangolin::Attach::Pix ( 175 ), 1.0, -640.0f/480.0f )
            .SetHandler ( new pangolin::Handler3D ( _scam ) );
}

void Viewer::Draw()
{
    glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    _dcam.Activate ( _scam );
    glClearColor ( 1.0f, 1.0f, 1.0f, 1.0f );
    // draw the key-frames and map points in memory


    for ( auto frame: Memory::_frames )
    {
        DrawPose ( frame.second->_T_c_w );
    }

    // temp pose
    for ( SE3& pose: _poses_tmp )
    {
        DrawPose ( pose );
    }

    // draw the map points
    for ( auto point : Memory::_points )
    {
        glPointSize ( 3 );
        glBegin ( GL_POINTS );
        glColor3d ( 0.1,0.1,0.1 );

        glVertex3d ( point.second->_pos_world[0], point.second->_pos_world[1], point.second->_pos_world[2] );
        glEnd();
    }

    pangolin::FinishFrame();
    usleep ( 1000 );
}

void Viewer::DrawPose ( const SE3& T_c_w )
{
    const static float w = 0.18;
    const static float h = w*0.75;
    const static float z = w*0.6;
    pangolin::OpenGlMatrix m;
    SE3 Twc = T_c_w.inverse();
    Eigen::Matrix4d T = Twc.matrix();
    m.m[0] = T ( 0,0 );
    m.m[1] = T ( 1,0 );
    m.m[2] = T ( 2,0 );
    m.m[3] = 0;

    m.m[4] = T ( 0,1 );
    m.m[5] = T ( 1,1 );
    m.m[6] = T ( 2,1 );
    m.m[7] = 0;

    m.m[8] = T ( 0,2 );
    m.m[9] = T ( 1,2 );
    m.m[10] = T ( 2,2 );
    m.m[11] = 0;

    m.m[12] = T ( 0,3 );
    m.m[13] = T ( 1,3 );
    m.m[14] = T ( 2,3 );
    m.m[15] = 1;

    // draw this camera

    glPushMatrix();
    glMultMatrixd ( m.m );

    glLineWidth ( 0.5f );
    glColor3f ( 0.6f,0.6f,0.6f );
    glBegin ( GL_LINES );
    glVertex3f ( 0,0,0 );
    glVertex3f ( w,h,z );
    glVertex3f ( 0,0,0 );
    glVertex3f ( w,-h,z );
    glVertex3f ( 0,0,0 );
    glVertex3f ( -w,-h,z );
    glVertex3f ( 0,0,0 );
    glVertex3f ( -w,h,z );

    glVertex3f ( w,h,z );
    glVertex3f ( w,-h,z );

    glVertex3f ( -w,h,z );
    glVertex3f ( -w,-h,z );

    glVertex3f ( -w,h,z );
    glVertex3f ( w,h,z );

    glVertex3f ( -w,-h,z );
    glVertex3f ( w,-h,z );
    glEnd();

    glPopMatrix();

}



}
