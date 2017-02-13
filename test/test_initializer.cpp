#include "ygz/Basic.h"
#include "ygz/Algorithm.h"
using namespace std; 
using namespace ygz; 
using namespace cv;

// 本程序测试初始化，使用仿真数据
// 路标点，位于z=2的平面上，用以测试
double landmarks[12][3] = 
{
    { -1, -1, 2 },
    { -1, 0, 2 },
    { -1, 1, 2 },
    { -1, 2, 2 },
    
    { 0, -1, 2 },
    { 0, 0, 2 },
    { 0, 1, 2 },
    { 0, 2, 2 },
    
    { 1, -1, 2 },
    { 1, 0, 2 },
    { 1, 1, 2 },
    { 1, 2, 2 },
};

// 相机数据使用配置文件的参数

int main( int argc, char** argv )
{
    ygz::Config::SetParameterFile("./config/default.yaml");
    // 两个帧的pose
    SE3 pose1 = SE3( 
        SO3::exp(Vector3d(0,0,0)),
        Vector3d(0,0,0)
    );
    SE3 pose2 = SE3( 
        SO3::exp(Vector3d(0,0,0)),
        Vector3d(1,0,0) // 这个在x方向有1平移
    );
    
    ygz::PinholeCamera* cam = new ygz::PinholeCamera(); 
    ygz::Frame::SetCamera( cam );
    ygz::Frame* frame1 = new ygz::Frame();
    ygz::Frame* frame2 = new ygz::Frame();
    
    cv::RNG rng;
    // 生成观测数据
    vector<Vector2d> px1, px2;
    px1.resize(12);
    px2.resize(12);
    for ( int i=0; i<12; i++ )
    {
        px1[i] = cam->World2Pixel( 
            Vector3d(landmarks[i][0], landmarks[i][1], landmarks[i][2]), pose1 )
            + Vector2d( rng.gaussian(1), rng.gaussian(1) )     // 加噪声 
        ;
        px2[i] = cam->World2Pixel( 
            Vector3d(landmarks[i][0], landmarks[i][1], landmarks[i][2]), pose2 )
            + Vector2d( rng.gaussian(1), rng.gaussian(1) )     // 加噪声 
        ;
    }
    
    LOG(INFO) << "Data generated, test initializer" <<endl;
    ygz::Initializer* init = new ygz::Initializer();
    
    bool ret = init->TryInitialize(
        px1, px2, 
        frame1, frame2
    );
    
    if ( ret )
        LOG(INFO)<<"Initialize succeeded."<<endl;
    else
        LOG(INFO)<<"Initialize failed."<<endl;
    
    delete cam; 
    delete frame1;
    delete frame2;
    delete init;
    return 0;
}