#ifndef FRAME_H_
#define FRAME_H_
#include "ygz/common_include.h"
#include "ygz/camera.h"
#include "ygz/map_point.h"

namespace ygz {
    
// Frame，帧
// 帧是一种数据对象，所以本身使用Struct，成员都使用public
struct Frame {
public:
    typedef shared_ptr<Frame> Ptr; // 通常传递Frame的指针来访问对象 
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    Frame() {}
    
    // Frames should be created by memeory, otherwise their ID may not be valid 
    Frame( const Frame& frame ) =delete; // Frame 由Memory管理，不允许复制
    
    // 构造函数，通常由Memroy建立，如果是临时对象，也可以由其他地方建立 
    // 如果是由其他地方建立的，通过Memroy::RegisterFrame添加到Memory中，这样它的id会合法化
    Frame( 
        const double& timestamp, 
        const SE3& T_c_w, 
        const bool is_keyframe, 
        const Mat& color, 
        const Mat& depth = Mat()
    ) : _timestamp(timestamp), _T_c_w(T_c_w), _is_keyframe(is_keyframe), _color(color), 
    _depth( depth ) {} 
    
    Frame operator = ( const Frame& f2 ) =delete; // 不允许赋值

    // called by system when reading parameters 
    static void SetCamera( PinholeCamera::Ptr camera ) {
        _camera = camera; 
    }
    
    // create the image pyramid, etc 
    void InitFrame();
    
    // return the camera position in the world 
    inline Vector3d Pos() const { return _T_c_w.inverse().translation(); }
    
    // check whether a point is in frame 
    inline bool InFrame( const Vector2d& pixel, const int& boarder = 10 ) const {
        return pixel[0] >= boarder && pixel[0] < _color.cols - boarder 
            && pixel[1] >= boarder && pixel[1] < _color.rows - boarder;
    }
    
    // 带level的查询
    inline bool InFrame( const Vector2d& pixel, const int& boarder, const int& level ) const {
        return pixel[0]/(1<<level) >= boarder && pixel[0]/(1<<level) < _color.cols - boarder 
            && pixel[1]/(1<<level) >= boarder && pixel[1]/(1<<level) < _color.rows - boarder;
    }
    
public:
    // data 
    // ID, 只有关键帧才拥有系统管理的id，可以直接通过id寻找到这个关键帧
    unsigned long _id   =0; 
    // 时间戳，暂时不用，在加入与速度相关的计算之后才会用到
    double  _timestamp  =0; 
    // 位姿，以T_C_W表示，C指Camera,W指World
    SE3     _T_c_w      =SE3();         // pose 
    // 关键帧标志
    bool    _is_keyframe    =false;     // 标识是否是关键帧
    
    // 关联的地图点，以id标识。只有向系统注册后的地图点才会有 id 
    list<unsigned long> _map_point; // associated map point 
    // 候选地图点，由特征提取算法给出
    vector<MapPoint>    _map_point_candidates; 
    
    // observed features
    // NOTE observations 与_map_point对应，即每个 map point 在这个帧上的投影位置
    // 对于关键帧，可以访问map point的_obs变量获取该位置，但对于非关键帧，由于memory中并没有记录非关键帧的信息，所以必须在这里访问
    // 在做 sparse alignment 的时候，也必须在这里访问像素位置
    // 这里的Vector3d，前两维为像素坐标，第三维是深度值
    // 如果深度值小于零，表示该观测是一个outlier
    list<Vector3d>    _observations;
    
    // 图像，原始的彩色图和深度图
    Mat     _color;     // if we have 
    Mat     _depth;     // if we have 
    
    // 金字塔，越往上越小，默认缩放倍数是2，因为2可以用SSE优化...虽然目前还没有用SSE
    int _pyramid_level;  // 层数，由config读取
    vector<Mat>  _pyramid;      // gray image pyramid, it must be CV_8U
    
    // camera, 静态对象，所有Frame共用一个 
    static PinholeCamera::Ptr   _camera;
    
    // 格子，可以用来提取关键点，有很多用途
    vector<int> _grid;        // grid occupancy 
    
public:
    // inner functions 
    // 建立金字塔
    void CreateImagePyramid();
    
};

}

#endif
