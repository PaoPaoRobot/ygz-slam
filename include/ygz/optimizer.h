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
        Frame::Ptr frame1, Frame::Ptr frame2,
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
    vector<bool> _visible_pts;
    Frame::Ptr _frame1 =nullptr, _frame2=nullptr;
    SE3 _TCR;   // estimated pose 
};

// pose estimation 
// 给定当前帧与匹配到的地图点，优化它的位姿
void FrameToMapBAPoseOnly( Frame::Ptr current, list<MatchPointCandidate>& candidates );

}
}

#endif
