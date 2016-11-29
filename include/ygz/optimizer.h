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
class SparseImgAlign {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW ;
    SparseImgAlign( SE3& TCR=SE3()) : _TCR(TCR) {}
    
    // 用稀疏直接法计算两个图像的运动，结果是T_21
    // 输入两个帧的指针以及金字塔的层数
    void SparseImageAlignmentCeres (
        Frame::Ptr frame1, Frame::Ptr frame2,
        const int& pyramid_level
    );
    
    SE3 GetEstimatedT21() const {return _TCR;}
    
protected:
    void precomputeReferencePatches(); 
    
protected:
    // data and parameters 
    bool _have_ref_patch =false;
    vector<PixelPattern> _patterns_ref;
    int _pyramid_level;
    double _scale;
    vector<bool> _visible_pts;
    Frame::Ptr _frame1, _frame2;
    SE3 _TCR;   // estimated pose 
};

}
}

#endif
