#include "ygz/utils.h"

namespace ygz
{

namespace utils
{

void GetWarpAffineMatrix (
    const Frame::Ptr& ref,
    const Frame::Ptr& curr,
    const Vector2d& px_ref,
    const Vector3d& pt_ref,
    const int& level,
    const SE3& T_c_r,
    Eigen::Matrix2d& A_cur_ref
)
{
    // LOG(INFO) << "ref TCW = \n "<<ref->_T_c_w.matrix()<<endl;
    // LOG(INFO) << "curr TCW = \n "<<curr->_T_c_w.matrix()<<endl;
    
    // Vector2d px_ref_ = ref->_camera->Camera2Pixel( pt_ref );
    // LOG(INFO) << "px ref is " << px_ref_.transpose()<<", and the input is "<<px_ref.transpose()<<endl; 
    
    // 像素上置一个偏移量
    Vector3d pt_ref_world = ref->_camera->Camera2World( pt_ref, ref->_T_c_w );
    
    // 偏移之后的3d点，深度取成和pt_ref一致
    Vector3d pt_du_ref = ref->_camera->Pixel2World ( px_ref + Vector2d ( WarpHalfPatchSize, 0 ), ref->_T_c_w, pt_ref[2] );
    Vector3d pt_dv_ref = ref->_camera->Pixel2World ( px_ref + Vector2d ( 0, WarpHalfPatchSize ), ref->_T_c_w, pt_ref[2] );
    
    // 让深度与偏移之前相等
    // pt_du_ref *= pt_ref[2] / pt_du_ref[2];
    // pt_dv_ref *= pt_ref[2] / pt_dv_ref[2];
    
    const Vector2d px_cur = curr->_camera->World2Pixel ( pt_ref_world, curr->_T_c_w );
    const Vector2d px_du = curr->_camera->World2Pixel ( pt_du_ref, curr->_T_c_w );
    const Vector2d px_dv = curr->_camera->World2Pixel ( pt_dv_ref, curr->_T_c_w );
    
    // 如果旋转不大，那么du应该接近( WarpHalfPatchSize, 0), dv亦然
    LOG(INFO) << "du = " << (px_du-px_cur).transpose() << endl;
    LOG(INFO) << "dv = " << (px_dv-px_cur).transpose() << endl;

    A_cur_ref.col ( 0 ) = ( px_du - px_cur ) / WarpHalfPatchSize;
    A_cur_ref.col ( 1 ) = ( px_dv - px_cur ) / WarpHalfPatchSize;
}

void WarpAffine (
    const Eigen::Matrix2d& A_c_r,
    const Mat& img_ref,
    const Vector2d& px_ref,
    const int& level_ref,
    const int& search_level,
    const int& half_patch_size,
    uint8_t* patch
)
{
    const int patch_size = half_patch_size*2;
    const Eigen::Matrix2d A_r_c = A_c_r.inverse();

    // Affine warp
    uint8_t* patch_ptr = patch;
    const Vector2d px_ref_pyr = px_ref / ( 1<<level_ref );
    for ( int y=0; y<patch_size; y++ )
    {
        for ( int x=0; x<patch_size; x++, ++patch_ptr )
        {
            Vector2d px_patch ( x-half_patch_size, y-half_patch_size );
            px_patch *= ( 1<<search_level );
            const Vector2d px ( A_r_c*px_patch + px_ref_pyr );
            if ( px[0]<0 || px[1]<0 || px[0]>=img_ref.cols-1 || px[1]>=img_ref.rows-1 )
                *patch_ptr = 0;
            else
                *patch_ptr = GetBilateralInterpUchar ( px[0], px[1], img_ref );
        }
    }
}

// TODO rewrite it into Ceres or call OpenCV's LK optical flow! 
bool Align2D (
    const Mat& cur_img,
    uint8_t* ref_patch_with_border,
    uint8_t* ref_patch,
    const int n_iter,
    Vector2d& cur_px_estimate,
    bool no_simd )
{
    const int halfpatch_size_ = 4;
    const int patch_size_ = 8;
    const int patch_area_ = 64;
    bool converged=false;

    // compute derivative of template and prepare inverse compositional
    float __attribute__ ( ( __aligned__ ( 16 ) ) ) ref_patch_dx[patch_area_];
    float __attribute__ ( ( __aligned__ ( 16 ) ) ) ref_patch_dy[patch_area_];
    Matrix3f H;
    H.setZero();

    // Gauss-Newton iteration 
    // compute gradient and hessian
    const int ref_step = patch_size_+2;
    float* it_dx = ref_patch_dx;
    float* it_dy = ref_patch_dy;
    for ( int y=0; y<patch_size_; ++y )
    {
        uint8_t* it = ref_patch_with_border + ( y+1 ) *ref_step + 1;
        for ( int x=0; x<patch_size_; ++x, ++it, ++it_dx, ++it_dy )
        {
            Eigen::Vector3f J;
            J[0] = 0.5 * ( it[1] - it[-1] );
            J[1] = 0.5 * ( it[ref_step] - it[-ref_step] );
            J[2] = 1;
            *it_dx = J[0];
            *it_dy = J[1];
            H += J*J.transpose();
        }
    }
    Matrix3f Hinv = H.inverse();
    float mean_diff = 0;

    // Compute pixel location in new image:
    float u = cur_px_estimate.x();
    float v = cur_px_estimate.y();

    // termination condition
    const float min_update_squared = 0.03*0.03;
    const int cur_step = cur_img.step.p[0];
//  float chi2 = 0;
    Eigen::Vector3f update;
    update.setZero();
    for ( int iter = 0; iter<n_iter; ++iter )
    {
        int u_r = floor ( u );
        int v_r = floor ( v );
        if ( u_r < halfpatch_size_ || v_r < halfpatch_size_ || u_r >= cur_img.cols-halfpatch_size_ || v_r >= cur_img.rows-halfpatch_size_ )
            break;

        if ( isnan ( u ) || isnan ( v ) ) // TODO very rarely this can happen, maybe H is singular? should not be at corner.. check
            return false;

        // compute interpolation weights
        float subpix_x = u-u_r;
        float subpix_y = v-v_r;
        float wTL = ( 1.0-subpix_x ) * ( 1.0-subpix_y );
        float wTR = subpix_x * ( 1.0-subpix_y );
        float wBL = ( 1.0-subpix_x ) *subpix_y;
        float wBR = subpix_x * subpix_y;

        // loop through search_patch, interpolate
        uint8_t* it_ref = ref_patch;
        float* it_ref_dx = ref_patch_dx;
        float* it_ref_dy = ref_patch_dy;
    //    float new_chi2 = 0.0;
        Eigen::Vector3f Jres;
        Jres.setZero();
        for ( int y=0; y<patch_size_; ++y )
        {
            uint8_t* it = ( uint8_t* ) cur_img.data + ( v_r+y-halfpatch_size_ ) *cur_step + u_r-halfpatch_size_;
            for ( int x=0; x<patch_size_; ++x, ++it, ++it_ref, ++it_ref_dx, ++it_ref_dy )
            {
                float search_pixel = wTL*it[0] + wTR*it[1] + wBL*it[cur_step] + wBR*it[cur_step+1];
                float res = search_pixel - *it_ref + mean_diff;
                Jres[0] -= res* ( *it_ref_dx );
                Jres[1] -= res* ( *it_ref_dy );
                Jres[2] -= res;
//        new_chi2 += res*res;
            }
        }
        
        update = Hinv * Jres;
        u += update[0];
        v += update[1];
        mean_diff += update[2];

#if SUBPIX_VERBOSE
        cout << "Iter " << iter << ":"
             << "\t u=" << u << ", v=" << v
             << "\t update = " << update[0] << ", " << update[1]
//         << "\t new chi2 = " << new_chi2 << endl;
#endif

             if ( update[0]*update[0]+update[1]*update[1] < min_update_squared )
        {
#if SUBPIX_VERBOSE
            cout << "converged." << endl;
#endif
            converged=true;
            break;
        }
    }

    cur_px_estimate << u, v;
    return converged;

}



}

}
