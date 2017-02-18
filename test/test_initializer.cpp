#include "ygz/Basic.h"
#include "ygz/Algorithm.h"

using namespace std; 
using namespace ygz; 
using namespace cv;

// 本程序测试初始化，使用仿真数据。测试initializer是否正常工作，以及init之后用一次Two view BA是否有精度提升。
// 路标点，位于z=2的平面上，用以测试H的情况
double landmarks_H[12][3] = 
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

// 路标点，位于z=2,3,4的不同层中，用以测试F的情况
double landmarks_F[12][3] = 
{
    { -1, -1, 2 },
    { -1, 1, 2 },
    { 1, -1, 2 },
    { 1, 1, 2 },
    
    { -1,-1, 3 },
    { -1, 1, 3 },
    { 1, -1, 3 },
    { 1,  1, 3 },
    
    { -1,-1, 4 },
    { -1, 1, 4 },
    { 1, -1, 4 },
    { 1,  1, 4 },
};

// 相机内参使用配置文件的参数

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
    vector<Vector2d> px1H, px2H;
    vector<Vector2d> px1F, px2F;
    px1H.resize(12);
    px2H.resize(12);
    px1F.resize(12);
    px2F.resize(12);
    
    for ( int i=0; i<12; i++ )
    {
        px1H[i] = cam->World2Pixel( 
            Vector3d(landmarks_H[i][0], landmarks_H[i][1], landmarks_H[i][2]), pose1 );
        
        px2H[i] = cam->World2Pixel( 
            Vector3d(landmarks_H[i][0], landmarks_H[i][1], landmarks_H[i][2]), pose2 );
        
        px1F[i] = cam->World2Pixel( 
            Vector3d(landmarks_F[i][0], landmarks_F[i][1], landmarks_F[i][2]), pose1 );
        
        px2F[i] = cam->World2Pixel( 
            Vector3d(landmarks_F[i][0], landmarks_F[i][1], landmarks_F[i][2]), pose2 );
        
        px1H[i] += Vector2d( rng.gaussian(2.0), rng.gaussian(2.0) );    // 加噪声 
        px2H[i] += Vector2d( rng.gaussian(2.0), rng.gaussian(2.0) );    // 加噪声 
        px1F[i] += Vector2d( rng.gaussian(2.0), rng.gaussian(2.0) );    // 加噪声 
        px2F[i] += Vector2d( rng.gaussian(2.0), rng.gaussian(2.0) );    // 加噪声 
    }
    
    LOG(INFO) << "Data generated, test initializer" <<endl;
    ygz::Initializer* init = new ygz::Initializer();
    
    bool ret = init->TryInitialize(
        px1F, px2F, 
        frame1, frame2
    );
    
    if ( ret )
        LOG(INFO)<<"Initialize succeeded."<<endl;
    else
        LOG(INFO)<<"Initialize failed."<<endl;
    
    LOG(INFO) << "T21 estimated by initializer = \n"<<init->GetT21().matrix()<<endl;
    
    frame2->_TCW = init->GetT21();
    vector<bool> inliers; 
    vector<Vector3d> pts_ref; 
    init->GetTriangluatedPoints( pts_ref, inliers );
    ba::TwoViewBACeres( frame1->_TCW, frame2->_TCW, px1F, px2F, inliers, pts_ref );
    
    // normalize the scale 
    double scale = frame2->_TCW.translation().norm();
    LOG(INFO) << "scale = "<<scale<<endl;
    frame2->_TCW.translation() = frame2->_TCW.translation()/scale;
    
    LOG(INFO) << "pose 2 true value = \n"<<pose2.matrix()<<"\nestimated = \n"<<frame2->_TCW.matrix()<<endl;
    
    for ( size_t i=0; i<pts_ref.size(); i++ ) 
    {
        LOG(INFO) << "map point "<<i<<"\ntrue value = "<<
            landmarks_F[i][0] << ", "<<
            landmarks_F[i][1] << ", "<<
            landmarks_F[i][2] << "\nestimated value = "<<pts_ref[i].transpose()/scale<<endl;
    }
    
    int cnt_inlier = 0;
    for ( bool in:inliers )
        if (in) cnt_inlier++;
        
    LOG(INFO) << "inliers: "<<cnt_inlier<<endl;
    
    delete cam; 
    delete frame1;
    delete frame2;
    delete init;
    return 0;
}