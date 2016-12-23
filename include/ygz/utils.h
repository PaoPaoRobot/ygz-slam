#ifndef YGZ_UTILS_H
#define YGZ_UTILS_H

#include "ygz/common_include.h"
#include "ygz/camera.h"

// 一些杂七杂八不知道放哪的东西
namespace ygz
{

class Frame;
class MapPoint;

namespace utils
{

// 三角化，给定两个feature和帧间变换，求深度
// 输入T12和1,2帧下的归一化相机坐标，输出第1帧下的三角化点
Vector3d TriangulateFeatureNonLin (
    const SE3& T,
    const Vector3d& feature1,
    const Vector3d& feature2
);

// 3d -> 2d projection
inline Vector2d Project2d ( const Vector3d& v )
{
    return v.head<2>() /v[2];
}

inline Vector3d ProjectHomo ( const Vector3d& v )
{
    return v/v[2];
}

// 稀疏直接法里用的pattern
enum {PATTERN_SIZE = 8};
const double PATTERN_DX[PATTERN_SIZE] = {0,-1,1,1,-2,0,2,0};
const double PATTERN_DY[PATTERN_SIZE] = {0,-1,-1,1,0,-2,0,2};

struct PixelPattern {
    float pattern[PATTERN_SIZE];
};

// 可能匹配到的地图点
struct MatchPointCandidate {
    unsigned long _observed_keyframe;
    unsigned long _map_point;
    Vector2d _keyframe_pixel;
    Vector2d _projected_pixel;
};

// 转换函数
inline Eigen::Vector2d Cv2Eigen ( const cv::Point2f& p )
{
    return Eigen::Vector2d ( p.x, p.y );
}

inline Eigen::Vector2d Cv2Eigen ( const cv::Point2d& p )
{
    return Eigen::Vector2d ( p.x, p.y );
}

// 双线性图像插值
// 假设图像是CV_8U的灰度图，返回/255之后的0～1之间的值
inline float GetBilateralInterp (
    const double& x, const double& y, const Mat& gray )
{
    const double xx = x - floor ( x );
    const double yy = y - floor ( y );
    uchar* data = &gray.data[ int ( y ) * gray.step + int ( x ) ];
    return float (
               ( 1-xx ) * ( 1-yy ) * data[0] +
               xx* ( 1-yy ) * data[1] +
               ( 1-xx ) *yy*data[ gray.step ] +
               xx*yy*data[gray.step+1]
           ) /255.0;
}

// 双线性图像插值
// 假设图像是CV_8U的灰度图，返回0～255间的灰度值，但可能会溢出?
inline uchar GetBilateralInterpUchar (
    const double& x, const double& y, const Mat& gray )
{
    const double xx = x - floor ( x );
    const double yy = y - floor ( y );
    uchar* data = &gray.data[ int ( y ) * gray.step + int ( x ) ];
    return uchar (
               ( 1-xx ) * ( 1-yy ) * data[0] +
               xx* ( 1-yy ) * data[1] +
               ( 1-xx ) *yy*data[ gray.step ] +
               xx*yy*data[gray.step+1]
           );
}

// 图像层面判断点是否落在里面
inline bool IsInside ( const Vector2d& pixel, cv::Mat& img, int boarder=10 )
{
    return pixel[0] >= boarder && pixel[0] < img.cols - boarder
           && pixel[1] >= boarder && pixel[1] < img.rows - boarder;
}

// 一些固定的雅可比
// xyz 到 相机坐标 的雅可比，平移在前
// 这里已经取了负号，不要再取一遍！
inline Eigen::Matrix<double,2,6> JacobXYZ2Cam ( const Vector3d& xyz )
{
    Eigen::Matrix<double,2,6> J;
    const double x = xyz[0];
    const double y = xyz[1];
    const double z_inv = 1./xyz[2];
    const double z_inv_2 = z_inv*z_inv;

    J ( 0,0 ) = -z_inv;           // -1/z
    J ( 0,1 ) = 0.0;              // 0
    J ( 0,2 ) = x*z_inv_2;        // x/z^2
    J ( 0,3 ) = y*J ( 0,2 );      // x*y/z^2
    J ( 0,4 ) = - ( 1.0 + x*J ( 0,2 ) ); // -(1.0 + x^2/z^2)
    J ( 0,5 ) = y*z_inv;          // y/z

    J ( 1,0 ) = 0.0;              // 0
    J ( 1,1 ) = -z_inv;           // -1/z
    J ( 1,2 ) = y*z_inv_2;        // y/z^2
    J ( 1,3 ) = 1.0 + y*J ( 1,2 ); // 1.0 + y^2/z^2
    J ( 1,4 ) = -J ( 0,3 );       // -x*y/z^2
    J ( 1,5 ) = -x*z_inv;         // x/z
    return J;
}

// xyz 到 像素坐标 的雅可比，平移在前
// 这里已经取了负号，不要再取一遍！
inline Eigen::Matrix<double,2,6> JacobXYZ2Pixel ( const Vector3d& xyz,  PinholeCamera* cam )
{
    Eigen::Matrix<double,2,6> J;
    const double x = xyz[0];
    const double y = xyz[1];
    const double z_inv = 1./xyz[2];
    const double z_inv_2 = z_inv*z_inv;

    J ( 0,0 ) = -z_inv*cam->fx();               // -fx/Z
    J ( 0,1 ) = 0.0;                            // 0
    J ( 0,2 ) = x*z_inv_2*cam->fx();            // fx*x/z^2
    J ( 0,3 ) = cam->fx() *y*J ( 0,2 );         // fx*x*y/z^2
    J ( 0,4 ) = -cam->fx() * ( 1.0 + x*J ( 0,2 ) ); // -fx*(1.0 + x^2/z^2)
    J ( 0,5 ) = cam->fx() *y*z_inv;             // fx*y/z

    J ( 1,0 ) = 0.0;                            // 0
    J ( 1,1 ) = -cam->fy() *z_inv;              // -fy/z
    J ( 1,2 ) = cam->fy() *y*z_inv_2;           // fy*y/z^2
    J ( 1,3 ) = cam->fy() * ( 1.0 + y*J ( 1,2 ) ); // fy*(1.0 + y^2/z^2)
    J ( 1,4 ) = -cam->fy() *x*J ( 1,2 );        // fy*-x*y/z^2
    J ( 1,5 ) = -cam->fy() *x*z_inv;            // fy*x/z

    return J;
}

// warp functions
static const int WarpHalfPatchSize = 4;      // half patch size
static const int WarpPatchSize = 8;      // patch size

// 计算 ref 与 current 之间的一个 affine warp
void GetWarpAffineMatrix (
    const Frame* ref,
    const Frame* curr,
    const Vector2d& px_ref,
    const Vector3d& pt_ref,
    const int & level,
    const SE3& T_c_r,
    Eigen::Matrix2d& A_cur_ref
);

// perform affine warp
void WarpAffine (
    const Eigen::Matrix2d& A_c_r,
    const cv::Mat& img_ref,
    const Vector2d& px_ref,
    const int& level_ref,
    const int& search_level,
    const int& half_patch_size,
    uint8_t* patch
);

// 计算最好的金字塔层数
// 选择一个分辨率，使得warp不要太大
inline int GetBestSearchLevel (
    const Eigen::Matrix2d& A_c_r,
    const int& max_level )
{
    int search_level = 0;
    double D = A_c_r.determinant();
    while ( D > 3.0 && search_level < max_level ) {
        search_level += 1;
        D *= 0.25;
    }
    return search_level;
}

// 寻找一个像素在图像中的匹配关系，类似于光流
// copy from SVO feature_alignment
// 注解：这个Align2D实现的有点问题，用g-n迭代，但是H矩阵是固定在ref的梯度上的，不会随着每次迭代更新
// 如果离正确匹配的距离较远，很可能有非凸性问题
// 最后一项参数用以判断是否根据收敛情况判断是否成功
bool Align2D (
    const cv::Mat& cur_img,
    uint8_t* ref_patch_with_border,
    uint8_t* ref_patch,
    const int n_iter,
    Vector2d& cur_px_estimate,
    bool convergence_condiiton = true
);

// Find a match by searching along the epipolar line without using any features.
// 需要假设 curr_frame 是已经知道 pose 的
bool FindEpipolarMatchDirect (
    const Frame* ref_frame,
    const Frame* cur_frame,
    const cv::KeyPoint& ref_ftr,
    const double& d_estimate,
    const double& d_min,
    const double& d_max,
    double& depth,
    Vector2d& matched_px
);

// 三角化计算特征点深度
inline bool DepthFromTriangulation (
    const SE3& T_search_ref,
    const Vector3d& f_ref,
    const Vector3d& f_cur,
    double& depth )
{
    Eigen::Matrix<double,3,2> A;
    A << T_search_ref.rotation_matrix() * f_ref, -f_cur;
    
    const Eigen::Matrix2d AtA = A.transpose() *A;
    
    // LOG(INFO) << "AtA determinant = " << AtA.determinant()<<endl;
    if ( AtA.determinant() < 1e-4 ) {
        // 行列式太小说明该方程解不稳定
        return false;
    }
    
    // LOG(INFO) << "A=\n"<<A<<endl;
    // LOG(INFO) << "TCR = \n" << T_search_ref.matrix()<<endl;
    // LOG(INFO) << "f_ref="<<f_ref.transpose()<<endl;
    // LOG(INFO) << "f_cur="<<f_cur.transpose()<<endl;
    
    // LOG(INFO) << "AtA = \n"<<AtA<<endl;
    
    Vector2d depth2 = - AtA.inverse() *A.transpose() *T_search_ref.translation();
    // LOG(INFO) << "depth2 = " << depth2.transpose() << endl;
    // Vector3d r = A*depth2;
    // LOG(INFO) << "Ad = " << r.transpose() << endl;
    depth = fabs ( depth2[0] );
    return true;
}

// SSE 计算块匹配方法
#if __SSE2__
// Horizontal sum of uint16s stored in an XMM register
inline int SumXMM_16 ( __m128i &target )
{
    unsigned short int sums_store[8];
    _mm_storeu_si128 ( ( __m128i* ) sums_store, target );
    return sums_store[0] + sums_store[1] + sums_store[2] + sums_store[3] +
           sums_store[4] + sums_store[5] + sums_store[6] + sums_store[7];
}
// Horizontal sum of uint32s stored in an XMM register
inline int SumXMM_32 ( __m128i &target )
{
    unsigned int sums_store[4];
    _mm_storeu_si128 ( ( __m128i* ) sums_store, target );
    return sums_store[0] + sums_store[1] + sums_store[2] + sums_store[3];
}
#endif

// SSD, from rpg_vikit
/// Zero Mean Sum of Squared Differences Cost
template<int HALF_PATCH_SIZE>
class ZMSSD
{
public:

    static const int patch_size_ = 2*HALF_PATCH_SIZE;
    static const int patch_area_ = patch_size_*patch_size_;
    static const int threshold_  = 2000*patch_area_;
    uint8_t* ref_patch_;
    int sumA_, sumAA_;

    ZMSSD ( uint8_t* ref_patch ) :
        ref_patch_ ( ref_patch ) {
        uint32_t sumA_uint=0, sumAA_uint=0;
        for ( int r = 0; r < patch_area_; r++ ) {
            uint8_t n = ref_patch_[r];
            sumA_uint += n;
            sumAA_uint += n*n;
        }
        sumA_ = sumA_uint;
        sumAA_ = sumAA_uint;
    }

    static int threshold() {
        return threshold_;
    }

    int computeScore ( uint8_t* cur_patch ) const {
        uint32_t sumB_uint = 0;
        uint32_t sumBB_uint = 0;
        uint32_t sumAB_uint = 0;
        for ( int r = 0; r < patch_area_; r++ ) {
            const uint8_t cur_pixel = cur_patch[r];
            sumB_uint  += cur_pixel;
            sumBB_uint += cur_pixel*cur_pixel;
            sumAB_uint += cur_pixel * ref_patch_[r];
        }
        const int sumB = sumB_uint;
        const int sumBB = sumBB_uint;
        const int sumAB = sumAB_uint;
        return sumAA_ - 2*sumAB + sumBB - ( sumA_*sumA_ - 2*sumA_*sumB + sumB*sumB ) /patch_area_;
    }

    // 神奇的 SSE .... 老子哪天也要学这个 SSE
    int computeScore ( uint8_t* cur_patch, int stride ) const {
        int sumB, sumBB, sumAB;
#if __SSE2__
        if ( patch_size_ == 8 ) {
            // From PTAM-GPL, Copyright 2008 Isis Innovation Limited
            __m128i xImageAsEightBytes;
            __m128i xImageAsWords;
            __m128i xTemplateAsEightBytes;
            __m128i xTemplateAsWords;
            __m128i xZero;
            __m128i xImageSums;   // These sums are 8xuint16
            __m128i xImageSqSums; // These sums are 4xint32
            __m128i xCrossSums;   // These sums are 4xint32
            __m128i xProduct;

            xImageSums = _mm_setzero_si128();
            xImageSqSums = _mm_setzero_si128();
            xCrossSums = _mm_setzero_si128();
            xZero = _mm_setzero_si128();

            uint8_t* imagepointer = cur_patch;
            uint8_t* templatepointer = ref_patch_;
            long unsigned int cur_stride = stride;

            xImageAsEightBytes=_mm_loadl_epi64 ( ( __m128i* ) imagepointer );
            imagepointer += cur_stride;
            xImageAsWords = _mm_unpacklo_epi8 ( xImageAsEightBytes,xZero );
            xImageSums = _mm_adds_epu16 ( xImageAsWords,xImageSums );
            xProduct = _mm_madd_epi16 ( xImageAsWords, xImageAsWords );
            xImageSqSums = _mm_add_epi32 ( xProduct, xImageSqSums );
            xTemplateAsEightBytes=_mm_load_si128 ( ( __m128i* ) templatepointer );
            templatepointer += 16;
            xTemplateAsWords = _mm_unpacklo_epi8 ( xTemplateAsEightBytes,xZero );
            xProduct = _mm_madd_epi16 ( xImageAsWords, xTemplateAsWords );
            xCrossSums = _mm_add_epi32 ( xProduct, xCrossSums );
            xImageAsEightBytes=_mm_loadl_epi64 ( ( __m128i* ) imagepointer );
            imagepointer += cur_stride;
            xImageAsWords = _mm_unpacklo_epi8 ( xImageAsEightBytes,xZero );
            xImageSums = _mm_adds_epu16 ( xImageAsWords,xImageSums );
            xProduct = _mm_madd_epi16 ( xImageAsWords, xImageAsWords );
            xImageSqSums = _mm_add_epi32 ( xProduct, xImageSqSums );
            xTemplateAsWords = _mm_unpackhi_epi8 ( xTemplateAsEightBytes,xZero );
            xProduct = _mm_madd_epi16 ( xImageAsWords, xTemplateAsWords );
            xCrossSums = _mm_add_epi32 ( xProduct, xCrossSums );

            xImageAsEightBytes=_mm_loadl_epi64 ( ( __m128i* ) imagepointer );
            imagepointer += cur_stride;
            xImageAsWords = _mm_unpacklo_epi8 ( xImageAsEightBytes,xZero );
            xImageSums = _mm_adds_epu16 ( xImageAsWords,xImageSums );
            xProduct = _mm_madd_epi16 ( xImageAsWords, xImageAsWords );
            xImageSqSums = _mm_add_epi32 ( xProduct, xImageSqSums );
            xTemplateAsEightBytes=_mm_load_si128 ( ( __m128i* ) templatepointer );
            templatepointer += 16;
            xTemplateAsWords = _mm_unpacklo_epi8 ( xTemplateAsEightBytes,xZero );
            xProduct = _mm_madd_epi16 ( xImageAsWords, xTemplateAsWords );
            xCrossSums = _mm_add_epi32 ( xProduct, xCrossSums );
            xImageAsEightBytes=_mm_loadl_epi64 ( ( __m128i* ) imagepointer );
            imagepointer += cur_stride;
            xImageAsWords = _mm_unpacklo_epi8 ( xImageAsEightBytes,xZero );
            xImageSums = _mm_adds_epu16 ( xImageAsWords,xImageSums );
            xProduct = _mm_madd_epi16 ( xImageAsWords, xImageAsWords );
            xImageSqSums = _mm_add_epi32 ( xProduct, xImageSqSums );
            xTemplateAsWords = _mm_unpackhi_epi8 ( xTemplateAsEightBytes,xZero );
            xProduct = _mm_madd_epi16 ( xImageAsWords, xTemplateAsWords );
            xCrossSums = _mm_add_epi32 ( xProduct, xCrossSums );

            xImageAsEightBytes=_mm_loadl_epi64 ( ( __m128i* ) imagepointer );
            imagepointer += cur_stride;
            xImageAsWords = _mm_unpacklo_epi8 ( xImageAsEightBytes,xZero );
            xImageSums = _mm_adds_epu16 ( xImageAsWords,xImageSums );
            xProduct = _mm_madd_epi16 ( xImageAsWords, xImageAsWords );
            xImageSqSums = _mm_add_epi32 ( xProduct, xImageSqSums );
            xTemplateAsEightBytes=_mm_load_si128 ( ( __m128i* ) templatepointer );
            templatepointer += 16;
            xTemplateAsWords = _mm_unpacklo_epi8 ( xTemplateAsEightBytes,xZero );
            xProduct = _mm_madd_epi16 ( xImageAsWords, xTemplateAsWords );
            xCrossSums = _mm_add_epi32 ( xProduct, xCrossSums );
            xImageAsEightBytes=_mm_loadl_epi64 ( ( __m128i* ) imagepointer );
            imagepointer += cur_stride;
            xImageAsWords = _mm_unpacklo_epi8 ( xImageAsEightBytes,xZero );
            xImageSums = _mm_adds_epu16 ( xImageAsWords,xImageSums );
            xProduct = _mm_madd_epi16 ( xImageAsWords, xImageAsWords );
            xImageSqSums = _mm_add_epi32 ( xProduct, xImageSqSums );
            xTemplateAsWords = _mm_unpackhi_epi8 ( xTemplateAsEightBytes,xZero );
            xProduct = _mm_madd_epi16 ( xImageAsWords, xTemplateAsWords );
            xCrossSums = _mm_add_epi32 ( xProduct, xCrossSums );

            xImageAsEightBytes=_mm_loadl_epi64 ( ( __m128i* ) imagepointer );
            imagepointer += cur_stride;
            xImageAsWords = _mm_unpacklo_epi8 ( xImageAsEightBytes,xZero );
            xImageSums = _mm_adds_epu16 ( xImageAsWords,xImageSums );
            xProduct = _mm_madd_epi16 ( xImageAsWords, xImageAsWords );
            xImageSqSums = _mm_add_epi32 ( xProduct, xImageSqSums );
            xTemplateAsEightBytes=_mm_load_si128 ( ( __m128i* ) templatepointer );
            templatepointer += 16;
            xTemplateAsWords = _mm_unpacklo_epi8 ( xTemplateAsEightBytes,xZero );
            xProduct = _mm_madd_epi16 ( xImageAsWords, xTemplateAsWords );
            xCrossSums = _mm_add_epi32 ( xProduct, xCrossSums );
            xImageAsEightBytes=_mm_loadl_epi64 ( ( __m128i* ) imagepointer );
            xImageAsWords = _mm_unpacklo_epi8 ( xImageAsEightBytes,xZero );
            xImageSums = _mm_adds_epu16 ( xImageAsWords,xImageSums );
            xProduct = _mm_madd_epi16 ( xImageAsWords, xImageAsWords );
            xImageSqSums = _mm_add_epi32 ( xProduct, xImageSqSums );
            xTemplateAsWords = _mm_unpackhi_epi8 ( xTemplateAsEightBytes,xZero );
            xProduct = _mm_madd_epi16 ( xImageAsWords, xTemplateAsWords );
            xCrossSums = _mm_add_epi32 ( xProduct, xCrossSums );

            sumB = SumXMM_16 ( xImageSums );
            sumAB = SumXMM_32 ( xCrossSums );
            sumBB = SumXMM_32 ( xImageSqSums );
        } else
#endif
        {
            uint32_t sumB_uint = 0;
            uint32_t sumBB_uint = 0;
            uint32_t sumAB_uint = 0;
            for ( int y=0, r=0; y < patch_size_; ++y ) {
                uint8_t* cur_patch_ptr = cur_patch + y*stride;
                for ( int x=0; x < patch_size_; ++x, ++r ) {
                    const uint8_t cur_px = cur_patch_ptr[x];
                    sumB_uint  += cur_px;
                    sumBB_uint += cur_px * cur_px;
                    sumAB_uint += cur_px * ref_patch_[r];
                }
            }
            sumB = sumB_uint;
            sumBB = sumBB_uint;
            sumAB = sumAB_uint;
        }
        return sumAA_ - 2*sumAB + sumBB - ( sumA_*sumA_ - 2*sumA_*sumB + sumB*sumB ) /patch_area_;
    }
};

typedef ZMSSD<4> PatchScore;  // 8x8 的 patch

}
}


#endif
