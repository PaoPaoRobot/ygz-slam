#ifndef YGZ_FRAME_H
#define YGZ_FRAME_H

#include "ygz/common_include.h"
#include "ygz/ORB/ORBVocabulary.h"

// for DBoW3 
#include "BowVector.h"
#include "FeatureVector.h"

namespace ygz {
    
// 前置声明
class PinholeCamera; 
class MapPoint;

// Frame，帧
// 帧是一种数据对象，所以本身使用Struct，成员都使用public

// TODO 关键帧的spanning tree
struct Frame {
public:
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
   
    ~Frame() {
        _obs.clear();
    }
    
    Frame operator = ( const Frame& f2 ) =delete; // 不允许赋值

    // called by system when reading parameters 
    static void SetCamera( PinholeCamera* camera ) {
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
    
    inline bool InFrame( const cv::Point2f& pixel, const int& boarder = 10 ) const {
        return pixel.x >= boarder && pixel.x < _color.cols - boarder 
            && pixel.y >= boarder && pixel.y < _color.rows - boarder;
    }
    
    // 带level的查询
    inline bool InFrame( const Vector2d& pixel, const int& boarder, const int& level ) const {
        return pixel[0]/(1<<level) >= boarder && pixel[0]/(1<<level) < _color.cols - boarder 
            && pixel[1]/(1<<level) >= boarder && pixel[1]/(1<<level) < _color.rows - boarder;
    }
    
    // 是否坏帧
    bool IsBad() const {
        return _bad;
    }
    
    // 计算观测到的地图点的平均深度和最小深度
    bool GetMeanAndMinDepth( double& mean_depth, double& min_depth ); 
    
    // 获得相机中心
    Vector3d GetCamCenter() const {
        return _T_c_w.inverse().translation();
    }
    
    // 得到共视关键最好的N个帧
    vector<Frame*> GetBestCovisibilityKeyframes( const int & N );
    
    // 判断某个地图点的投影是否在视野内，同时检查其夹角
    bool IsInFrustum( MapPoint* mp, float viewingCosLimit=0.5 );
    
    // 添加 Covisibility 连接
    // 权重为共视点数量
    void AddConnection( Frame* kf, const int& weight );
    
    // 更新 Covisibility 和 Essential 关系
    void UpdateConnections(); 
    
    // 更新共视帧的情况，根据 _connected_keyframe_weights 计算
    void UpdateBestCovisibles(); 
    
    // 获得所有特征点的描述子
    cv::Mat GetAllDescriptors(); 
    
    // 将备选点的描述转换成 bow
    void ComputeBoW();
    
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
    
    // 2D特征点，由特征提取算法给出
    vector<cv::KeyPoint>    _map_point_candidates; 
    vector<Mat>             _descriptors;       // 每个特征点的描述，由ORB计算
    
    // 观测
    // observations 与_map_point对应，即每个 map point 在这个帧上的投影位置
    // 对于关键帧，可以访问map point的_obs变量获取该位置，但对于非关键帧，由于memory中并没有记录非关键帧的信息，所以必须在这里访问
    // 在做 sparse alignment 的时候，也必须在这里访问像素位置
    // 这里的Vector3d，前两维为像素坐标，第三维是深度值
    // 如果深度值小于零，表示该观测是一个outlier
    map<unsigned long, Vector3d, std::less<unsigned long>, Eigen::aligned_allocator<Vector3d>> _obs;
    
    // 图像，原始的彩色图和深度图
    Mat     _color;     // if we have 
    Mat     _depth;     // if we have 
    
    // 金字塔，越往上越小，默认缩放倍数是2，因为2可以用SSE优化...虽然目前还没有用SSE
    int _pyramid_level;  // 层数，由config读取
    vector<Mat>  _pyramid;      // gray image pyramid, it must be CV_8U
    
    // camera, 静态对象，所有Frame共用一个 
    static PinholeCamera*   _camera;
    
    // 格子，可以用来提取关键点，有很多用途
    vector<int> _grid;        // grid occupancy 
    
    // 参考的关键帧
    Frame* _ref_keyframe   =nullptr; 
    
    bool _bad=false;  // bad flag 
    
    // 共视的关键帧
    map<Frame*, int> _connected_keyframe_weights;
    // 排序后的结果，从大到小
    vector<Frame*> _cov_keyframes;      // 按照权重排序后的关键帧
    vector<int> _cov_weights;           // 排序后的权重
    
    // DBoW 
    DBoW3::BowVector _bow_vec;
    DBoW3::FeatureVector _feature_vec;
    
    static ORBVocabulary* _vocab;
    
public:
    // inner functions 
    // 建立金字塔
    void CreateImagePyramid();
    
};

}

#endif
