#ifndef YGZ_CVUTILS_H_
#define YGZ_CVUTILS_H_

// 和图像处理相关的一些工具函数，例如图像插值，三角化等等

#include "ygz/Basic/Common.h"

namespace ygz 
{
namespace cvutils 
{
    
// *************************************************************************************
// 三角化
// 给定ref和current中的归一化坐标，计算二者的深度
// 行列式太小说明该方程解不稳定，返回false
// 请注意返回的深度可能是负值
inline bool DepthFromTriangulation (
    const SE3& T_search_ref,
    const Vector3d& f_ref,
    const Vector3d& f_cur,
    double& depth1, 
    double& depth2, 
    const double& determinant_th=1e-4
)
{
    Eigen::Matrix<double,3,2> A;
    A << T_search_ref.rotation_matrix() * f_ref, -f_cur;
    Eigen::Matrix2d AtA = A.transpose() *A;
    if ( AtA.determinant() < determinant_th ) {
        return false;
    }
    Vector2d depth2d = - AtA.inverse() *A.transpose() *T_search_ref.translation();
    depth1 = depth2d[0];
    depth2 = depth2d[1];
    return true;
}
// *************************************************************************************
// 图像插值

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
    
// *************************************************************************************
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

// *************************************************************************************
// 图像配准
// Alignment using Ceres
// TODO 对Ceres的效率深表怀疑
/**
 * @brief Align an image patch(in ref) with the current image
 * @param[in] cur_img The current image 
 * @param[in] ref_patch the patch in reference frame, by default is 64x64
 * @param[in] ref_patch_with_boarder the patch with boarder, used to compute the gradient (or FEJ)
 * @param[out] cur_px_estimate the estimated position in current image, must have an initial value
 * @return True if successful
 */
bool Align2DCeres( 
    const Mat& cur_img, 
    uint8_t* ref_patch, 
    uint8_t* ref_patch_with_boarder, 
    Vector2d& cur_px_estimate 
);


}
}

#endif // YGZ_CVUTILS_H_