#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "ygz/utils.h"
#include "ygz/map_point.h"
#include "ygz/frame.h"
#include "ygz/ceres_types.h"


namespace ygz
{

namespace utils
{

Vector3d TriangulateFeatureNonLin (
    const SE3& T,
    const Vector3d& feature1,
    const Vector3d& feature2
)
{
    Vector3d f2 = T.rotation_matrix() * feature2;
    Vector2d b;
    b[0] = T.translation().dot ( feature1 );
    b[1] = T.translation().dot ( f2 );
    Eigen::Matrix2d A;
    A ( 0,0 ) = feature1.dot ( feature1 );
    A ( 1,0 ) = feature1.dot ( f2 );
    A ( 0,1 ) = -A ( 1,0 );
    A ( 1,1 ) = -f2.dot ( f2 );
    Vector2d lambda = A.inverse() * b;
    Vector3d xm = lambda[0] * feature1;
    Vector3d xn = T.translation() + lambda[1] * f2;
    return ( xm + xn ) /2;
}

void GetWarpAffineMatrix (
    const Frame* ref,
    const Frame* curr,
    const Vector2d& px_ref,
    const Vector3d& pt_ref,
    const int& level,
    const SE3& T_c_r,
    Eigen::Matrix2d& A_cur_ref
)
{
    // 像素上置一个偏移量
    Vector3d pt_ref_world = ref->_camera->Camera2World ( pt_ref, ref->_T_c_w );

    // 偏移之后的3d点，深度取成和pt_ref一致
    Vector3d pt_du_ref = ref->_camera->Pixel2World ( px_ref + Vector2d ( WarpHalfPatchSize, 0 ) * ( 1<<level ), ref->_T_c_w, pt_ref[2] );
    Vector3d pt_dv_ref = ref->_camera->Pixel2World ( px_ref + Vector2d ( 0, WarpHalfPatchSize ) * ( 1<<level ), ref->_T_c_w, pt_ref[2] );

    const Vector2d px_cur = curr->_camera->World2Pixel ( pt_ref_world, curr->_T_c_w );
    const Vector2d px_du = curr->_camera->World2Pixel ( pt_du_ref, curr->_T_c_w );
    const Vector2d px_dv = curr->_camera->World2Pixel ( pt_dv_ref, curr->_T_c_w );

    // 如果旋转不大，那么du应该接近( WarpHalfPatchSize, 0), dv亦然
    // LOG(INFO) << "du = " << (px_du-px_cur).transpose() << endl;
    // LOG(INFO) << "dv = " << (px_dv-px_cur).transpose() << endl;

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
            {
                *patch_ptr = 0;
            }
            else
            {
                *patch_ptr = GetBilateralInterpUchar ( px[0], px[1], img_ref );
            }
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
    bool convergence_condition )
{
    const int halfpatch_size_ = 4;
    const int patch_size_ = 8;
    const int patch_area_ = 64;
    bool converged=false;
    Vector2d first_estimate = cur_px_estimate;

    // compute derivative of template and prepare inverse compositional
    float __attribute__ ( ( __aligned__ ( 16 ) ) ) ref_patch_dx[patch_area_];
    float __attribute__ ( ( __aligned__ ( 16 ) ) ) ref_patch_dy[patch_area_];

    // Hessian should be estimated at current instead of ref?
    Matrix3f H;
    H.setZero();

    // Gauss-Newton iteration
    // compute gradient and hessian
    int ref_step = patch_size_+2;
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
    float min_update_squared = 0.001;
    int cur_step = cur_img.cols;
    Eigen::Vector3f update;
    update.setZero();
    int iter = 0;
    vector<float> chi2_vec;

    LOG(INFO)<<"start interation ... "<<endl;
    bool error_increased = false;
    for ( ; iter<n_iter; ++iter )
    {
        float chi2 = 0;
        int u_r = floor ( u );
        int v_r = floor ( v );
        if ( u_r < halfpatch_size_ || v_r < halfpatch_size_ || u_r >= cur_img.cols-halfpatch_size_ || v_r >= cur_img.rows-halfpatch_size_ )
        {
            // LOG(INFO) << "u_r = "<<u_r <<", v_r = " << v_r <<endl;
            break;
        }

        if ( isnan ( u ) || isnan ( v ) ) // TODO very rarely this can happen, maybe H is singular? should not be at corner.. check
        {
            return false;
        }

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

                chi2 += res*res;
            }
        }

        update = Hinv * Jres;
        u += update[0];
        v += update[1];
        mean_diff += update[2];

        if ( iter>0 && chi2 > chi2_vec.back() ) // error increased
        {
            error_increased = true;
            break;
        }

        chi2_vec.push_back ( chi2 );

        if ( update[0]*update[0]+update[1]*update[1] < min_update_squared )
        {
            first_estimate<<u,v;
            converged=true;
            break;
        }
    }

    cur_px_estimate << u, v;

    LOG(INFO) << update.transpose() << endl;
    if ( convergence_condition )
        return converged;

    if ( converged == true )
    {
        LOG(INFO) << "accepted, converged."<<endl;
        return true;
    }
    if ( converged == false )
    {
        // return false;
        // LOG(INFO) << "iter = "<<iter<<", update = " << update.transpose()<<endl;
        // for ( float& c: chi2_vec )
        // LOG(INFO) << c ;

        // 没有收敛，可能出现误差上升，或者达到最大迭代次数
        if ( chi2_vec.empty() )
        {
            LOG ( INFO ) << "rejected because u,v runs outside."<<endl;
            return false;
        }
        /*
        LOG ( INFO ) << "chi2 = " ;
        for ( auto chi2:chi2_vec )
            LOG ( INFO ) << chi2;
        */

        if ( error_increased )
        {
            if ( chi2_vec.back() <15000 )
            {
                cur_px_estimate = first_estimate;
                LOG(INFO)<<"return true"<<endl;
                return true;
            }
            LOG(INFO)<<"error increased, return false, chi2 = "<<chi2_vec.back()<<endl;
            return false;
        }

        if ( chi2_vec.back() <15000 ) {
            LOG(INFO) << "return true" << endl;
            return true;
        }
        
        LOG(INFO) << "return false"<<endl;
        return false;
    }
    LOG(INFO)<<"return false"<<endl;
    return false;
}

bool Align2DCeres( 
    const Mat& cur_img, 
    uint8_t* ref_patch, 
    Vector2d& cur_px_estimate 
)
{
    // LOG(INFO) << "initial estimate: " << cur_px_estimate.transpose()<<endl;
    ceres::Problem problem;
    Vector2d px = cur_px_estimate;
    CeresAlignmentError* p = new CeresAlignmentError(
        ref_patch, cur_img
    );
    
    problem.AddResidualBlock(
        p, 
        nullptr,
        cur_px_estimate.data()
    );
    
    ceres::Solver::Options options;
    options.max_num_iterations = 10;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    ceres::Solver::Summary summary;
    ceres::Solve( options, &problem, &summary );
    
    // LOG(INFO) << summary.final_cost << endl;
    bool bad = false;
    if ( (px-cur_px_estimate).norm() > 5 )
        bad = true;
    if ( summary.final_cost > 2000 ) 
        bad = true;
    if ( bad == false ) 
        return true; 
    
    // LOG(INFO)<<"retrying"<<endl;
    // 尝试不使用FeJ重新算一遍
    cur_px_estimate = px; 
    p->SetFej( false );
    ceres::Solve( options, &problem, &summary );
    // LOG(INFO) << summary.final_cost << endl;
    
    if ( (px-cur_px_estimate).norm() > 5 )
        return false;
    if ( summary.final_cost > 2000 ) 
        return false;
    return true;
}

bool FindEpipolarMatchDirect (
    const Frame* ref_frame,
    const Frame* cur_frame,
    const cv::KeyPoint& ref_ftr,
    const double& d_estimate,
    const double& d_min,
    const double& d_max,
    double& depth,
    Vector2d& matched_px
)
{
    const int halfpatch_size = 4;
    const int patch_size = 8;

    SE3 T_cur_ref = cur_frame->_T_c_w * ref_frame->_T_c_w.inverse();
    int zmssd_best = PatchScore::threshold();
    Vector2d uv_best;

    // Compute start and end of epipolar line in old_kf for match search, on unit plane!
    Vector3d pt_ref = ref_frame->_camera->Pixel2Camera ( utils::Cv2Eigen ( ref_ftr.pt ) );
    Vector2d A = Project2d ( T_cur_ref * ( pt_ref*d_min ) );    // 相机坐标
    Vector2d B = Project2d ( T_cur_ref * ( pt_ref*d_max ) );
    Vector2d ep_dir = A - B; // epipolar direction

    // Compute affine warp matrix
    Eigen::Matrix2d A_cur_ref;
    utils::GetWarpAffineMatrix (
        ref_frame, cur_frame, utils::Cv2Eigen ( ref_ftr.pt ), pt_ref*d_estimate, ref_ftr.octave, T_cur_ref,  A_cur_ref
    );

    // LOG ( INFO ) << "A_c_r = " << A_cur_ref << endl;

    // feature pre-selection
    bool reject = false;

    /* 现在没有 edgelet
    if ( ref_ftr.type == Feature::EDGELET && options_.epi_search_edgelet_filtering )
    {
        const Vector2d grad_cur = ( A_cur_ref_ * ref_ftr.grad ).normalized();
        const double cosangle = fabs ( grad_cur.dot ( epi_dir_.normalized() ) );
        if ( cosangle < options_.epi_search_edgelet_max_angle )
        {
            reject_ = true;
            return false;
        }
    }
    */

    int search_level = GetBestSearchLevel ( A_cur_ref, 2 );
    // LOG ( INFO ) <<"feature level = " << ref_ftr.octave << endl;
    // LOG ( INFO ) <<"searched level = " << search_level << endl;

    // 匹配局部地图用的 patch
    uchar patch[halfpatch_size*halfpatch_size];
    // 带边界的，左右各1个像素
    uchar patch_with_border[ ( patch_size+2 ) * ( patch_size+2 )];

    // Find length of search range on epipolar line
    Vector2d px_A ( cur_frame->_camera->Camera2Pixel ( Vector3d ( A[0], A[1], 1 ) ) );
    Vector2d px_B ( cur_frame->_camera->Camera2Pixel ( Vector3d ( B[0], B[1], 1 ) ) );
    double epi_length = ( px_A-px_B ).norm() / ( 1<<search_level );

    Vector2d px_ref = utils::Cv2Eigen ( ref_ftr.pt );

    /*
    // show the epipolar line in current
    cv::Mat curr_show = cur_frame->_color.clone();
    cv::Mat ref_show = ref_frame->_color.clone();

    cv::circle( ref_show, cv::Point2f(px_ref[0], px_ref[1]), 5, cv::Scalar(0,250,0), 2 );
    cv::line( curr_show, cv::Point2f(px_A[0], px_A[1]), cv::Point2f(px_B[0], px_B[1]), cv::Scalar(0,250,0), 2 );

    cv::imshow("px in ref", ref_show );
    cv::imshow("epi line in curr", curr_show );
    cv::waitKey(1);
    */

    // Warp reference patch at ref_level
    WarpAffine ( A_cur_ref, ref_frame->_pyramid[ref_ftr.octave], px_ref,
                 ref_ftr.octave, search_level, halfpatch_size+1, patch_with_border );

    /*
    cv::Mat ref_patch ( WarpPatchSize+2, WarpPatchSize+2, CV_8UC1 );
    for ( size_t i=0; i<WarpPatchSize+2; i++ )
        for ( size_t j=0; j<WarpPatchSize+2; j++ )
        {
            ref_patch.ptr<uchar> ( i ) [j] = patch_with_border[ i* ( WarpPatchSize+2 ) + j];
        }

    cv::namedWindow ( "warpped ref patch", CV_WINDOW_NORMAL );
    cv::imshow ( "warpped ref patch", ref_patch );
    cv::resizeWindow ( "warpped ref patch", 500, 500 );

    cv::Rect2d rect_ref (
        ref_ftr.pt.x/ ( 1<<ref_ftr.octave ) -WarpHalfPatchSize,
        ref_ftr.pt.y/ ( 1<<ref_ftr.octave )-WarpHalfPatchSize,
        WarpPatchSize,
        WarpPatchSize
    );
    cv::Mat real_ref_patch = ref_frame->_pyramid[ref_ftr.octave] ( rect_ref ).clone();
    cv::namedWindow ( "real ref patch", CV_WINDOW_NORMAL );
    cv::imshow ( "real ref patch", real_ref_patch );
    cv::resizeWindow ( "real ref patch", 500, 500 );

    cv::waitKey ( 0 );
    */
    
    uint8_t* ref_patch_ptr = patch;

    for ( int y=1; y<patch_size+1; ++y, ref_patch_ptr += patch_size )
    {
        uint8_t* ref_patch_border_ptr = patch_with_border + y* ( patch_size+2 ) + 1;
        for ( int x=0; x<patch_size; ++x )
        {
            ref_patch_ptr[x] = ref_patch_border_ptr[x];
        }
    }

    Vector2d px_cur;

    // 如果极线很短，说明这个像素基本收敛，此时尝试用2D alignment
    if ( epi_length < 2.0 )
    {
        px_cur = ( px_A+px_B ) /2.0;
        Vector2d px_scaled ( px_cur/ ( 1<<search_level ) );
        bool res;
        // LOG ( INFO ) << "epipolar line is short! " << endl;
        res = Align2D (
                  cur_frame->_pyramid[search_level], patch_with_border, patch,
                  10, px_scaled, false );

        /*
        cv::Rect2d rect (
            px_scaled[0] -WarpHalfPatchSize,
            px_scaled[1] -WarpHalfPatchSize,
            WarpPatchSize,
            WarpPatchSize
        );

        cv::Mat matched_patch = cur_frame->_pyramid[search_level] ( rect ).clone();
        cv::namedWindow ( "curr patch", CV_WINDOW_NORMAL );
        cv::imshow ( "curr patch", matched_patch );
        cv::resizeWindow ( "curr patch", 500, 500 );
        cv::waitKey ( 0 );
        */

        if ( res )
        {
            LOG(INFO) << "align 2d succeed"<<endl;
            px_cur = px_scaled* ( 1<<search_level );
            double depth2;
            if ( DepthFromTriangulation (
                        T_cur_ref, pt_ref,
                        cur_frame->_camera->Pixel2Camera ( px_cur ), depth, depth2 ) )
            {
                LOG(INFO) << "estimated depth = " << depth <<endl;

                /*
                cv::Mat curr_show = cur_frame->_color.clone();
                cv::Mat ref_show = ref_frame->_color.clone();

                LOG(INFO) << "stange depth from align2D = " << depth << endl;

                cv::circle( ref_show, cv::Point2f(px_ref[0], px_ref[1]), 3, cv::Scalar(0,250,0), 1 );
                cv::line( curr_show, cv::Point2f(px_A[0], px_A[1]), cv::Point2f(px_B[0], px_B[1]), cv::Scalar(0,250,0), 1 );
                cv::circle( curr_show, cv::Point2f(px_cur[0], px_cur[1]), 3, cv::Scalar(0,250,0), 1 );
                cv::imshow("epi line in curr", curr_show );
                cv::imshow("epi line in ref", ref_show );

                cv::waitKey(0);
                */

                matched_px = px_cur;
                LOG(INFO) << "return true." <<endl;
                return true;
            }
            matched_px = px_cur;
            // LOG ( INFO ) << "rejected by triagulation"<<endl;
            return false;
        }
        else
        {
            matched_px = px_cur;
            LOG ( INFO ) << "rejected by align 2d"<<endl;
            return false;
        }
    }

    // 极线比较长，此时沿着极线搜索之
    size_t n_steps = epi_length/0.7; // one step per pixel
    Vector2d step = ep_dir/n_steps;

    if ( n_steps > 1000 )
    {
        //LOG ( WARNING ) << "epipolar search with too many steps"<<endl;
        return false;
    }

    // for matching, precompute sum and sum2 of warped reference patch
    int pixel_sum = 0;
    int pixel_sum_square = 0;
    PatchScore patch_score ( patch );

    // now we sample along the epipolar line
    Vector2d uv = B-step;
    Eigen::Vector2i last_checked_pxi ( 0,0 );
    ++n_steps;
    for ( size_t i=0; i<n_steps; ++i, uv+=step )
    {
        Vector2d px ( cur_frame->_camera->Camera2Pixel ( Vector3d ( uv[0],uv[1],1 ) ) );
        Eigen::Vector2i pxi ( px[0]/ ( 1<<search_level ) +0.5,
                              px[1]/ ( 1<<search_level ) +0.5 ); // +0.5 to round to closest int
        if ( pxi == last_checked_pxi )
        {
            continue;
        }
        last_checked_pxi = pxi;

        // check if the patch is full within the new frame
        if ( !cur_frame->InFrame ( pxi.cast<double>(), patch_size, search_level ) )
        {
            continue;
        }

        // TODO interpolation would probably be a good idea
        uint8_t* cur_patch_ptr = cur_frame->_pyramid[search_level].data
                                 + ( pxi[1]-halfpatch_size ) *cur_frame->_pyramid[search_level].cols
                                 + ( pxi[0]-halfpatch_size );
        int zmssd = patch_score.computeScore ( cur_patch_ptr, cur_frame->_pyramid[search_level].cols );

        if ( zmssd < zmssd_best )
        {
            zmssd_best = zmssd;
            uv_best = uv;
        }
    }

    if ( zmssd_best < PatchScore::threshold() )
    {
        px_cur = cur_frame->_camera->Camera2Pixel ( Vector3d ( uv_best[0], uv_best[1], 1 ) );
        /*
        cv::circle( curr_show, cv::Point2f(px_cur[0], px_cur[1]), 5, cv::Scalar(0,250,0), 2 );
        cv::imshow("epi line in curr", curr_show );
        cv::waitKey(1);
        */

        // bool res;
        /*
        if ( options_.align_1d )
            res = feature_alignment::align1D (
                      cur_frame.img_pyr_[search_level_], ( px_A-px_B ).cast<float>().normalized(),
                      patch_with_border_, patch_, options_.align_max_iter, px_scaled, h_inv_ );
        */

        // 居然又align一遍，别align了行不？
        // LOG ( INFO ) << "epipolar line is long!"<<endl;
        Vector2d px_scaled ( px_cur/ ( 1<<search_level ) );
        bool res = Align2D (
                       cur_frame->_pyramid[search_level], patch_with_border, patch,
                       10, px_scaled, false );

        /*
        cv::Rect2d rect (
            px_scaled[0] -WarpHalfPatchSize,
            px_scaled[1] -WarpHalfPatchSize,
            WarpPatchSize,
            WarpPatchSize
        );

        cv::Mat matched_patch = cur_frame->_pyramid[search_level] ( rect ).clone();
        cv::namedWindow ( "curr patch", CV_WINDOW_NORMAL );
        cv::imshow ( "curr patch", matched_patch );
        cv::resizeWindow ( "curr patch", 500, 500 );
        cv::waitKey ( 0 );
        */

        if ( res )
        {
            px_cur = px_scaled* ( 1<<search_level );
            double depth2;

            if ( DepthFromTriangulation (
                        T_cur_ref, pt_ref,
                        cur_frame->_camera->Pixel2Camera ( px_cur ), depth, depth2 ) )
            {
                if ( depth > 10 )
                {
                    /*
                    cv::Mat curr_show = cur_frame->_color.clone();
                    cv::Mat ref_show = ref_frame->_color.clone();

                    LOG(INFO) << "stange depth = " << depth << endl;
                    cv::circle( ref_show, cv::Point2f(px_ref[0], px_ref[1]), 3, cv::Scalar(0,250,0), 1 );
                    cv::line( curr_show, cv::Point2f(px_A[0], px_A[1]), cv::Point2f(px_B[0], px_B[1]), cv::Scalar(0,250,0), 1 );
                    cv::circle( curr_show, cv::Point2f(px_cur[0], px_cur[1]), 3, cv::Scalar(0,250,0), 1 );
                    cv::imshow("epi line in curr", curr_show );
                    cv::imshow("epi line in ref", ref_show );
                    cv::waitKey(0);
                    */
                }
                LOG ( INFO ) << "epipolar search succeed"<<endl;
                // LOG ( INFO ) << "estimated depth = " << depth << endl;
                /*

                cv::Mat ref_show = ref_frame->_color.clone();
                cv::Mat curr_show = cur_frame->_color.clone();
                cv::circle( ref_show, cv::Point2f(px_ref[0], px_ref[1]), 2, cv::Scalar(0,250,0), 2 );
                cv::circle( curr_show, cv::Point2f(px_cur[0], px_cur[1]), 2, cv::Scalar(0,250,0), 2 );
                cv::imshow("depth filter ref", ref_show );
                cv::imshow("depth filter curr", curr_show );
                cv::waitKey(0);
                */

                matched_px = px_cur;
                return true;
            }
            LOG ( INFO ) << "rejected by triagulation"<<endl;
            matched_px = px_cur;
            return false;
        }
        LOG ( INFO ) << "rejected by align 2d"<<endl;
        matched_px = px_cur;
        return false;
    }

    // LOG ( INFO ) << "reject because no best matched patch. "<<endl;
    matched_px = px_cur;
    return false;
}



}

}
