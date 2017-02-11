#ifndef YGZ_FRAME_H_
#define YGZ_FRAME_H_

#include "ygz/Basic/Common.h"


namespace ygz {
    
// 前置声明
    
class PinholeCamera; 
struct MapPoint;
struct Feature;

// Frame，帧
// 帧是一种基本数据对象，本身使用Struct，成员都使用public
// 一个帧拥有多个点特征，有些点特征能够三角化成地图点（单目），有些则是等待三角化的2D点
// 它们默认以ORB形式提取，只在关键帧处提取

struct Frame {
    
    struct Option {
        int _pyramid_level =3;
    } _option;
    
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    Frame() {}
    
    // Frames should be created by memeory, otherwise their ID may not be valid 
    Frame( const Frame& frame ) =delete; // Frame 由Memory管理，不允许复制
    Frame operator = ( const Frame& f2 ) =delete; // 不允许赋值
    
    ~Frame();

    // called by system when reading parameters 
    static void SetCamera( PinholeCamera* camera ) 
    {
        _camera = camera; 
    }
    
    // create the image pyramid, etc 
    void InitFrame();
    
    // return the camera position in the world 
    inline Vector3d Pos() const { return _TCW.inverse().translation(); }
    
    // check whether a point is in frame 
    inline bool InFrame( const Vector2d& pixel, const int& boarder = 10 ) const 
    {
        return pixel[0] >= boarder && pixel[0] < _color.cols - boarder 
            && pixel[1] >= boarder && pixel[1] < _color.rows - boarder;
    }
    
    inline bool InFrame( const cv::Point2f& pixel, const int& boarder = 10 ) const 
    {
        return pixel.x >= boarder && pixel.x < _color.cols - boarder 
            && pixel.y >= boarder && pixel.y < _color.rows - boarder;
    }
    
    // 带level的查询
    inline bool InFrame( const Vector2d& pixel, const int& boarder, const int& level ) const 
    {
        return pixel[0]/(1<<level) >= boarder && pixel[0]/(1<<level) < _color.cols - boarder 
            && pixel[1]/(1<<level) >= boarder && pixel[1]/(1<<level) < _color.rows - boarder;
    }
    
    // 计算观测到的地图点的平均深度和最小深度
    bool GetMeanAndMinDepth( double& mean_depth, double& min_depth ); 
    
    // 获得相机中心
    Vector3d GetCamCenter() const 
    {
        return _TCW.inverse().translation();
    }
    
    // 得到共视关键最好的N个帧
    vector<Frame*> GetBestCovisibilityKeyframes( const int & N=10 );
    
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
    
    // 设置字典
    static void SetORBVocabulary( ORBVocabulary* orb_vocab )
    {
        _vocab = orb_vocab;
    }
    
    // 清理特征点
    void CleanAllFeatures(); 
    
    // data 
    // ID，每个帧都有
    unsigned long _id   =0; 
    
    // 关键帧 id 
    unsigned long _keyframe_id  =0;
    
    // 时间戳，暂时不用，在加入与速度相关的计算之后才会用到
    double  _timestamp  =0; 
    
    // 位姿，以T_C_W表示， C 指 Camera, W 指 World
    SE3     _TCW =SE3();         // pose 
    
    // 关键帧标志
    bool    _is_keyframe    =false;     // 标识是否是关键帧
    
    // 观测到的 Feature
    // 可能按照已经三角化和未三角化来分开存放,可以减少一些不必要的遍历
    vector<Feature*>      _features;
    
    // 图像，原始的彩色图和深度图
    Mat     _color;     // if we have 
    Mat     _depth;     // if we have 
    
    // 金字塔，越往上越小，默认缩放倍数是2，因为2可以用SSE优化...虽然目前还没有用SSE
    vector<Mat>  _pyramid;      // gray image pyramid, it must be CV_8U
    
    // camera, 静态对象，所有Frame共用一个 
    static PinholeCamera*   _camera;
    
    // 参考的关键帧
    Frame* _ref_keyframe   =nullptr; 
    
    bool _bad   =false;  // bad flag 
    
    // 共视的关键帧
    map<Frame*, int> _connected_keyframe_weights;
    
    // 排序后的结果，从大到小
    vector<Frame*> _cov_keyframes;      // 按照权重排序后的关键帧
    vector<int> _cov_weights;           // 排序后的权重
    
    // DBoW 
    DBoW3::BowVector _bow_vec;
    DBoW3::FeatureVector _feature_vec;
    
    // 字典，共有一个
    static ORBVocabulary* _vocab;
    
    // inner functions 
    // 建立金字塔
    void CreateImagePyramid();
    
};

}

#endif // YGZ_FRAME_H_
