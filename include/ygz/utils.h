#ifndef YGZ_UTILS_H
#define YGZ_UTILS_H

#include "ygz/common_include.h"
#include "ygz/camera.h"
#include "ygz/frame.h"

// 一些杂七杂八不知道放哪的东西 
namespace ygz {
    
namespace utils {
    
// 稀疏直接法里用的pattern
enum {PATTERN_SIZE = 8};
const double PATTERN_DX[PATTERN_SIZE] ={0,-1,1,1,-2,0,2,0};
const double PATTERN_DY[PATTERN_SIZE] ={0,-1,-1,1,0,-2,0,2};

struct PixelPattern 
{
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
inline Eigen::Vector2d Cv2Eigen( const cv::Point2f& p ) {
    return Eigen::Vector2d( p.x, p.y );
}

inline Eigen::Vector2d Cv2Eigen( const cv::Point2d& p ) {
    return Eigen::Vector2d( p.x, p.y );
}

// 双线性图像插值
// 假设图像是CV_8U的灰度图，返回/255之后的0～1之间的值
inline float GetBilateralInterp( 
    const double& x, const double& y, const Mat& gray ) {
    const double xx = x - floor ( x );
    const double yy = y - floor ( y );
    uchar* data = &gray.data[ int(y) * gray.step + int(x) ];
    return float (
        ( 1-xx ) * ( 1-yy ) * data[0] +
        xx* ( 1-yy ) * data[1] +
        ( 1-xx ) *yy*data[ gray.step ] +
        xx*yy*data[gray.step+1]
    )/255.0;
}

// 双线性图像插值
// 假设图像是CV_8U的灰度图，返回0～255间的灰度值，但可能会溢出?
inline uchar GetBilateralInterpUchar( 
    const double& x, const double& y, const Mat& gray ) {
    const double xx = x - floor ( x );
    const double yy = y - floor ( y );
    uchar* data = &gray.data[ int(y) * gray.step + int(x) ];
    return uchar (
        ( 1-xx ) * ( 1-yy ) * data[0] +
        xx* ( 1-yy ) * data[1] +
        ( 1-xx ) *yy*data[ gray.step ] +
        xx*yy*data[gray.step+1]
    );
}

inline bool IsInside( const Vector2d& pixel, cv::Mat& img, int boarder=10 ) {
    return pixel[0] >= boarder && pixel[0] < img.cols - boarder 
        && pixel[1] >= boarder && pixel[1] < img.rows - boarder;
}

// 一些固定的雅可比
// xyz 到 相机坐标 的雅可比，平移在前
// 这里已经取了负号，不要再取一遍！
inline Eigen::Matrix<double,2,6> JacobXYZ2Cam( const Vector3d& xyz ) 
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
inline Eigen::Matrix<double,2,6> JacobXYZ2Pixel( const Vector3d& xyz,  PinholeCamera::Ptr cam ) 
{
    Eigen::Matrix<double,2,6> J;
    const double x = xyz[0];
    const double y = xyz[1];
    const double z_inv = 1./xyz[2];
    const double z_inv_2 = z_inv*z_inv;
    
    J ( 0,0 ) = -z_inv*cam->fx();               // -fx/Z
    J ( 0,1 ) = 0.0;                            // 0
    J ( 0,2 ) = x*z_inv_2*cam->fx();            // fx*x/z^2
    J ( 0,3 ) = cam->fx()*y*J ( 0,2 );          // fx*x*y/z^2
    J ( 0,4 ) = -cam->fx()*( 1.0 + x*J(0,2));   // -fx*(1.0 + x^2/z^2)
    J ( 0,5 ) = cam->fx()*y*z_inv;              // fx*y/z

    J ( 1,0 ) = 0.0;                            // 0
    J ( 1,1 ) = -cam->fy()*z_inv;               // -fy/z
    J ( 1,2 ) = cam->fy()*y*z_inv_2;            // fy*y/z^2
    J ( 1,3 ) = cam->fy()*(1.0 + y*J ( 1,2 ));  // fy*(1.0 + y^2/z^2)
    J ( 1,4 ) = -cam->fy()*x*J ( 1,2 );         // fy*-x*y/z^2
    J ( 1,5 ) = -cam->fy()*x*z_inv;             // fy*x/z
    
    return J;
}

// warp functions 
static const int WarpHalfPatchSize = 5;      // half patch size 
static const int WarpPatchSize = 10;      // half patch size 
    
// 计算 ref 与 current 之间的一个 affine warp 
void GetWarpAffineMatrix( 
    const Frame::Ptr& ref, 
    const Frame::Ptr& curr,
    const Vector2d& px_ref,
    const Vector3d& pt_ref, 
    const int & level, 
    const SE3& T_c_r,
    Eigen::Matrix2d& A_cur_ref
);
    
// perform affine warp 
void WarpAffine( 
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
inline int GetBestSearchLevel( 
    const Eigen::Matrix2d& A_c_r, 
    const int& max_level )
{
    int search_level = 0;
    double D = A_c_r.determinant();
    while ( D > 3.0 && search_level < max_level )
    {
        search_level += 1;
        D *= 0.25;
    }
    return search_level;
}

// 寻找一个像素在图像中的匹配关系，类似于光流
// copy from SVO feature_alignment
bool Align2D( 
    const cv::Mat& cur_img,
    uint8_t* ref_patch_with_border,
    uint8_t* ref_patch,
    const int n_iter,
    Vector2d& cur_px_estimate,
    bool no_simd = false
);



}
}


#endif