#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

#include "ygz/common_include.h"
#include "ygz/g2o_types.h"
#include "ygz/ceres_types.h"
#include "ygz/frame.h"
#include "ygz/utils.h"

using namespace ygz::utils;

namespace ygz
{

class LocalMapping;
namespace opti
{

// two view BA, used in intialization 
void TwoViewBAG2O (
    const unsigned long& frameID1,
    const unsigned long& frameID2
);

// two view BA, used in intialization 
void TwoViewBACeres (
    const unsigned long& frameID1,
    const unsigned long& frameID2
);

// pose only BA
// 在普通帧寻找匹配点之后调用
void OptimizePoseCeres(
    Frame* current,
    bool robust = false
);

// sparse image alignment, using ceres 
// 这里模仿 DSO 的做法，不是计算4x4的patch，而是算一个8维的pattern
// TODO 能否用SSE加速？
class SparseImgAlign {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW ;
    SparseImgAlign( const SE3& TCR=SE3()) : _TCR(TCR) {}
    
    // 用稀疏直接法计算两个图像的运动，结果是T_21
    // 输入两个帧的指针以及金字塔的层数
    void SparseImageAlignmentCeres (
        Frame* frame1, Frame* frame2,
        const int& pyramid_level
    );
    
    // 获得T21，即TCR的值
    SE3 GetEstimatedT21() const {return _TCR;}
    
    // 预设一个TCR值
    void SetTCR( const SE3& TCR ) { _TCR = TCR; }
    
protected:
    // 计算参考帧中的块
    void PrecomputeReferencePatches(); 
    
protected:
    // data and parameters 
    bool _have_ref_patch =false;
    vector<PixelPattern> _patterns_ref;
    int _pyramid_level;
    double _scale;
    Frame* _frame1 =nullptr, _frame2=nullptr;
    SE3 _TCR;   // estimated pose 
};


// Depth Filter 
// 想了半天，还是先用着，要是不好用再改别的
// Depth filter是一个高斯－均匀混合滤波器，与单纯高斯分布，可以区分 inlier 和 outlier
// 线程部分重新设计了一下，原理保持不变
struct Seed {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    static int batch_counter;
    static int seed_counter;
    
    int batch_id;                //!< Batch id is the id of the keyframe for which the seed was created.
    unsigned long frame_id;      // 最先提取这个seed的frame
    int id;                      //!< Seed ID, only used for visualization.
    MapPoint* ftr;                //!< Feature in the keyframe for which the depth should be computed.
    float a;                     //!< a of Beta distribution: When high, probability of inlier is large.
    float b;                     //!< b of Beta distribution: When high, probability of outlier is large.
    float mu;                    //!< Mean of normal distribution.
    float z_range;               //!< Max range of the possible depth.
    float sigma2;                //!< Variance of normal distribution.
    Eigen::Matrix2d patch_cov;          //!< Patch covariance in reference image.
    Seed ( const unsigned long& frame, MapPoint* ftr, float depth_mean, float depth_min );
    
    inline void PrintInfo() {
        LOG(INFO) << " mu = " << mu << ", sigma2 = "<< sigma2 << sigma2 << ", a = "<<a<<", b = "<<b <<endl;
    }
};

// 我究竟干了些什么。。。除了把下划线挪到前面之外。。。
// 注意 depth filter 实际上是在用普通帧的信息去更新关键帧
class DepthFilter 
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    DepthFilter( LocalMapping* local_mapping ):
        _local_mapping(local_mapping) {}
    
    // 这东西就别拷来拷去了
    DepthFilter( const DepthFilter& ) =delete;  
    DepthFilter& operator = ( const DepthFilter& ) =delete; 
    
    /// Depth-filter config parameters
    struct Options {
        bool check_ftr_angle;                       //!< gradient features are only updated if the epipolar line is orthogonal to the gradient.
        bool epi_search_1d;                         //!< restrict Gauss Newton in the epipolar search to the epipolar line.
        bool verbose;                               //!< display output.
        bool use_photometric_disparity_error;       //!< use photometric disparity error instead of 1px error in tau computation.
        int max_n_kfs;                              //!< maximum number of keyframes for which we maintain seeds.
        double sigma_i_sq;                          //!< image noise.
        double seed_convergence_sigma2_thresh;      //!< threshold on depth uncertainty for convergence.
        Options()
            : check_ftr_angle ( false ),
              epi_search_1d ( false ),
              verbose ( false ),
              use_photometric_disparity_error ( false ),
              max_n_kfs ( 3 ),
              sigma_i_sq ( 5e-4 ),
              seed_convergence_sigma2_thresh ( 150.0 )
        {}
    } _options;
    
    
    /// Add frame to the queue to be processed.
    void AddFrame ( Frame* frame );

    /// Add new keyframe to the queue
    void AddKeyframe ( Frame* frame, double depth_mean, double depth_min );

    /// Remove all seeds which are initialized from the specified keyframe. This
    /// function is used to make sure that no seeds points to a non-existent frame
    /// when a frame is removed from the map.
    void RemoveKeyframe ( Frame* frame );

    // 一些平凡的函数
    /// If the map is reset, call this function such that we don't have pointers
    /// to old frames.
    void reset();

    /// Returns a copy of the seeds belonging to frame. Thread-safe.
    /// Can be used to compute the Next-Best-View in parallel.
    /// IMPORTANT! Make sure you hold a valid reference counting pointer to frame
    /// so it is not being deleted while you use it.
    void GetSeedsCopy ( Frame* frame, std::list<Seed>& seeds );

    /// Return a reference to the seeds. This is NOT THREAD SAFE!
    std::list<Seed, Eigen::aligned_allocator<Seed> >& GetSeeds() {
        return _seeds;
    }

    /// Bayes update of the seed, x is the measurement, tau2 the measurement uncertainty
    static void UpdateSeed (
        const float& x,
        const float& tau2,
        Seed* seed );

    /// Compute the uncertainty of the measurement.
    static double ComputeTau (
        const SE3& T_ref_cur,
        const Vector3d& f,
        const double& z,
        const double& px_error_angle );
    
protected:
    std::list<Seed, Eigen::aligned_allocator<Seed> > _seeds;
    double _new_keyframe_min_depth =0.0;       //!< Minimum depth in the new keyframe. Used for range in new seeds.
    double _new_keyframe_mean_depth=0.0;      //!< Maximum depth in the new keyframe. Used for range in new seeds.
    bool _new_keyframe_set =false;
    deque<Frame*>   _frame_queue;    // 帧队列，在多线程中有用
    LocalMapping* _local_mapping =nullptr;
    
    /// Initialize new seeds from a frame.
    void InitializeSeeds ( Frame::Ptr frame );

    /// Update all seeds with a new measurement frame.
    virtual void UpdateSeeds ( Frame::Ptr frame );

    /// When a new keyframe arrives, the frame queue should be cleared.
    void ClearFrameQueue();

    /// A thread that is continuously updating the seeds.
    void UpdateSeedsLoop();
};

}
}

#endif
