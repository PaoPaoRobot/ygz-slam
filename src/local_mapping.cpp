#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "ygz/local_mapping.h"
#include "ygz/memory.h"
#include "ygz/utils.h"
#include "ygz/optimizer.h"

using namespace ygz::utils;

namespace ygz {
    
LocalMapping::LocalMapping()
{
    _num_local_keyframes = Config::get<int>("LocalMapping.local_keyframes");
    _num_local_map_points = Config::get<int>("LocalMapping.local_mappoints");
    _image_width = Config::get<int>("image.width");
    _image_height = Config::get<int>("image.height");
    _cell_size = Config::get<int>("feature.cell");
    _grid_rows = ceil(double(_image_height)/_cell_size);
    _grid_cols = ceil(double(_image_width)/_cell_size);
    _pyramid_level = Config::get<int>("frame.pyramid");
}

void LocalMapping::AddKeyFrame ( Frame::Ptr keyframe )
{
    assert( keyframe->_is_keyframe==true );
    // _local_keyframes.push_back( keyframe->_id );
    _local_keyframes.insert( keyframe->_id );
    
    // TODO 考虑把新增的keyframe中的地图点存储起来，并去掉视野外的点
    for ( unsigned long& point_id: keyframe->_map_point ) {
        MapPoint::Ptr map_point = Memory::GetMapPoint( point_id );
        if ( map_point->_bad ) continue;
        _local_map_points.insert( point_id );
    }
    
    // 删除太远的地图点和 key frame
    if ( _local_keyframes.size() > _num_local_keyframes ) {
        auto iter_min = std::min_element( _local_keyframes.begin(), _local_keyframes.end() );
        _local_keyframes.erase( iter_min );
    }
    
    // 如果某个地图点不在所有的key-frame中出现，则删除之？还是说只要出了当前帧就删除之？
    // Local map的地图点太少，可能会影响追踪性能，因为普通帧是没有新特征点的
    if ( _local_map_points.size() > _num_local_map_points ) {
        // 删除一些旧的
    }
    
    // call Local Bundle Adjustment
    // LocalBA( keyframe );
}

void LocalMapping::AddMapPoint ( const long unsigned int& map_point_id )
{
    assert( Memory::GetMapPoint(map_point_id) != nullptr );
    _local_map_points.insert( map_point_id );
}


// 寻找地图与当前帧之间的匹配，当前帧需要有位姿的粗略估计
// 这一步有点像光流
// 希望得到的匹配点能在图像中均匀分布，而不要扎堆在一起，所以使用网格进行区分
// 如果匹配数量足够，调用一次仅有pose的优化
bool LocalMapping::TrackLocalMap ( Frame::Ptr current )
{
    // Step 1
    // 建立匹配点的网格，提候选点
    LOG(INFO) << "frames in local map: "<< this->_local_keyframes.size()<< endl;
    LOG(INFO) << "map points in local map: "<< this->_local_map_points.size() << endl;
    
    _grid.clear();
    _grid.resize( _grid_rows*_grid_cols );
    
    int cntCandidate =0;
    for ( auto it = _local_map_points.begin(); it!=_local_map_points.end(); it++ ) {
        MapPoint::Ptr map_point = Memory::GetMapPoint( *it );
        if ( map_point->_bad == true )
            continue;
        // 检查这个地图点是否可见
        Vector3d pt_curr = current->_camera->World2Camera( map_point->_pos_world, current->_T_c_w );
        if ( pt_curr[2] < 0 ) // 在相机后面
            continue; 
        Vector2d px_curr = current->_camera->Camera2Pixel( pt_curr );
        if ( !current->InFrame(px_curr) )  // 不在图像中
            continue; 
        
        // 检测该顶点的 observation 里，有没有在相邻关键帧的
        MatchPointCandidate candidate;
        
        for ( auto& obs_pair: map_point->_obs ) {
            unsigned long keyframe_id = obs_pair.first;
            for ( auto it=_local_keyframes.begin(); it!=_local_keyframes.end(); it++ ) {
                if ( *it == keyframe_id ) {
                    candidate._observed_keyframe = *it;
                    candidate._map_point = map_point->_id;
                    candidate._keyframe_pixel = obs_pair.second.head<2>();
                    candidate._projected_pixel = px_curr;
                    // break;
                    
                    int k = static_cast<int> ( px_curr[1]/_cell_size ) *_grid_cols
                                + static_cast<int> ( px_curr[0]/_cell_size );
                    cntCandidate ++;
                    _grid[k].push_back( candidate );
                }
            }
        }
    }
    
    LOG(INFO) << "Find total "<<cntCandidate <<" candidates."<<endl;
    
    // Step 2 
    // 寻找每个候选点和当前帧的匹配
    
    int cntSuccess = 0;
    int cntFailed = 0;
    
    set<unsigned long> matched_points; 
    for ( size_t i=0; i<_grid.size(); i++ ) {
        for ( size_t j=0; j<_grid[i].size(); j++ ) {
            // 类似于光流的匹配 
            MatchPointCandidate candidate = _grid[i][j];
            Vector2d matched_px = candidate._projected_pixel; 
            bool success = TestDirectMatch( current, candidate, matched_px );
            if ( !success ) {  // 没有匹配到
                
                /*
                // 我要看一下怎么样的匹配才算失败
                Mat img_ref = Memory::GetFrame(candidate._observed_keyframe)->_color.clone();
                Mat img_curr = current->_color.clone();
                cv::circle( img_ref, cv::Point2f( candidate._keyframe_pixel[0], candidate._keyframe_pixel[1]), 5, cv::Scalar(0,0,250), 2);
                cv::circle( img_curr, cv::Point2f( matched_px[0], matched_px[1]), 5, cv::Scalar(0,0,250), 2);
                cv::imshow("wrong match ref", img_ref );
                cv::imshow("wrong match curr", img_curr );
                cv::waitKey(0);
                */
                
                cntFailed++;
                continue;
            } 
            
            /*
            Mat img_ref = Memory::GetFrame(candidate._observed_keyframe)->_color.clone();
            Mat img_curr = current->_color.clone();
            cv::circle( img_ref, cv::Point2f( candidate._keyframe_pixel[0], candidate._keyframe_pixel[1]), 5, cv::Scalar(0,250,0), 2);
            cv::circle( img_curr, cv::Point2f( matched_px[0], matched_px[1]), 5, cv::Scalar(0,250,0), 2);
            cv::imshow("correct match ref", img_ref );
            cv::imshow("correct match curr", img_curr );
            cv::waitKey(0);
            */
            
            if ( matched_points.find(candidate._map_point) == matched_points.end() ) { // 这个地图点没有被匹配过
                // 在current frame里增加一个对该地图点的观测，以便将来使用
                current->_map_point.push_back( candidate._map_point );
                // 匹配到的点作为观测值
                current->_observations.push_back( Vector3d(matched_px[0], matched_px[1], 1) );
                matched_points.insert( candidate._map_point );
                cntSuccess++;
                break; // 如果这个map point已经被匹配过了，就没必要再匹配了
            }
        }
    } 
    
    LOG(INFO) << "success = " << cntSuccess << ", failed = "<<cntFailed << endl; 
    
    if ( current->_map_point.size() < 10 ) { // TODO magic number, adjust it!
        // 匹配不够
        LOG(WARNING) << "insufficient matched pixels, abort this frame ... "<< endl;
        return false;
    }
        
    // Step 3
    // 根据前面的匹配信息，计算当前帧的Pose，以及优化这些地图点的位姿
    // Call Local bundle adjustment to optimize the current pose and map point 
    opti::OptimizePoseCeres( current, true );
    // remove the outliers by reprojection 
    auto iter_obs = current->_observations.begin();
    auto iter_pt = current->_map_point.begin();
    
    int cnt_outlier = 0;
    for ( ; iter_obs!=current->_observations.end();  ) {
        MapPoint::Ptr map_point = Memory::GetMapPoint( *iter_pt );
        Vector3d pt = current->_camera->World2Camera( map_point->_pos_world, current->_T_c_w);
        Vector2d px = current->_camera->Camera2Pixel( pt );
        Vector2d obs = iter_obs->head<2>();
        double error = (px-obs).norm();
        // LOG(INFO) << "reprojection error = "<< error << endl;
        
        if ( error > 10 ) { // magic number again! 
            // 重投影误差太大，认为这是个外点
            // TODO 想清楚这里是直接去掉呢，还是按照重投影来计算？
            // 如果计算重投影，那么由于该点被遮挡，它的纹理可能和map point处的纹理非常不同，这会使得local mapping在匹配模板时失败
            // 但是，直接去掉的话，又可能导致当前帧观测的数量太少，后一帧跟不上？
            
            // reset 也有点问题，可能把本该看不见的东西变成可见了？
            // 由于被遮挡，该点的深度就会发生改变，导致和其他特征点出现不一致
            
            iter_obs = current->_observations.erase( iter_obs );
            iter_pt = current->_map_point.erase( iter_pt );
            // (*iter_obs) [2] = -1;
            // iter_obs++;
            // iter_pt++;
            
            cnt_outlier++;
            continue; 
        } else {
            (*iter_obs)[2] = pt[2];
            iter_obs++;
            iter_pt++;
        }
    }
    
    LOG(INFO) << "outliers = " << cnt_outlier << endl;
    
    // Step 4
    // 现在当前帧应该全是内点了吧，再优化一次以求更精确
    // 这里需要使用局部地图的 ba 
    LocalBA( current );
    // 把观测按照重投影位置设置一下
    // 这里设重投影位置待议，由于误差存在，可能重投影位置不对，而之前至少是模板匹配得来的
    
    iter_obs = current->_observations.begin();
    iter_pt = current->_map_point.begin();
    
    // 看看重投影和obs有何不同
    cv::Mat img_show = current->_color.clone();
    for ( ; iter_obs!=current->_observations.end(); iter_obs++, iter_pt++ ) {
        // reset the observation 
        MapPoint::Ptr map_point = Memory::GetMapPoint( *iter_pt );
        Vector3d pt = current->_camera->World2Camera( map_point->_pos_world, current->_T_c_w );
        Vector2d px = current->_camera->World2Pixel( map_point->_pos_world, current->_T_c_w );
        
        cv::circle( img_show, cv::Point2f( (*iter_obs)[0], (*iter_obs)[1] ), 5, cv::Scalar(0,0,250), 2 );
        cv::circle( img_show, cv::Point2f( px[0], px[1] ), 5, cv::Scalar(0,250,0), 2 );
        
        // iter_obs->head<2>() = px;
        
        // 在地图点中添加额外观测
        if ( map_point->_converged == false )
            map_point->_extra_obs.push_back( ExtraObservation( pt/pt[2], current->_T_c_w) );
        (*iter_obs)[2] = pt[2]; // 只重设一下距离试试？
        
    }
    cv::imshow("obs vs reproj" , img_show);
    cv::waitKey(0);
    
    LOG(INFO) << "final observations: "<<current->_observations.size()<<endl;
    
    return true;
}

void LocalMapping::LocalBA ( Frame::Ptr current )
{
    ceres::Problem problem;
    vector<Vector6d*> poses;  
    
    Vector6d* pose = new Vector6d();
    Vector3d r = current->_T_c_w.so3().log(), t=current->_T_c_w.translation();
    pose->head<3>() = t;
    pose->tail<3>() = r;
    poses.push_back( pose );
    
    // 把每个帧与current 有关的map point拿出来的优化？还是说所有的放在一起优化？
    auto iter_mp = current->_map_point.begin();
    auto iter_obs = current->_observations.begin();
    
    map<unsigned long, Vector3d> mp_backup; // 缓存优化前的地图点坐标
    
    for ( ; iter_mp!=current->_map_point.end(); iter_mp++, iter_obs++ ) {
        MapPoint::Ptr mp = Memory::GetMapPoint( *iter_mp );
        
        Vector3d pt_curr = current->_camera->Pixel2Camera( iter_obs->head<2>() );
        if ( mp->_converged ) {
            // 对于收敛的地图点，不再优化它的位置，只给位姿估计提供信息
            problem.AddResidualBlock (
                new ceres::AutoDiffCostFunction<CeresReprojectionErrorPoseOnly,2,6> (
                    new CeresReprojectionErrorPoseOnly ( pt_curr.head<2>(), mp->_pos_world ) 
                ),
                nullptr,
                pose->data()
            );
        } else {
            // 未收敛的地图点，同时优化点的位置
            // 这是当前帧观测到的地图点，添加一个当前帧与它的pose-point对
            mp_backup[mp->_id] = mp->_pos_world;
            
            problem.AddResidualBlock (
                new ceres::AutoDiffCostFunction<CeresReprojectionError,2,6,3> (
                    new CeresReprojectionError ( pt_curr.head<2>() )
                ),
                nullptr,
                pose->data(),
                mp->_pos_world.data()
            );
            
            // 同时，这个point又被许多别的帧看到，也要添加在那些帧里的观测
            for ( auto& obs_pair : mp->_obs ) {
                Frame::Ptr obs_frame = Memory::GetFrame( obs_pair.first );
                Vector3d pt_obs = obs_frame->_camera->Pixel2Camera( obs_pair.second.head<2>() );
                // 但是不希望优化那些关键帧的位姿，使用 structure only 
                problem.AddResidualBlock (
                    new ceres::AutoDiffCostFunction<CeresReprojectionErrorPointOnly,2,3> (
                        new CeresReprojectionErrorPointOnly ( pt_obs.head<2>(), obs_frame->_T_c_w )
                    ),
                    nullptr,
                    mp->_pos_world.data()
                );
            }
            
            for ( auto& extra_obs : mp->_extra_obs ) {
                problem.AddResidualBlock (
                    new ceres::AutoDiffCostFunction<CeresReprojectionErrorPointOnly,2,3> (
                        new CeresReprojectionErrorPointOnly ( extra_obs._pt.head<2>(), extra_obs._TCW )
                    ),
                    nullptr,
                    mp->_pos_world.data()
                );
            }
        }
    }
    
    // Optimize it! 
    // 解之
    
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    // options.linear_solver_type = ceres::SPARSE_SCHUR;
    // options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve ( options, &problem, &summary );
    // cout<< summary.FullReport() << endl;
    
    current->_T_c_w = SE3( SO3::exp(pose->tail<3>()), pose->head<3>() );
    delete pose;
    
    // 判断地图点是否收敛
    for ( auto& backup_pair: mp_backup ) {
        MapPoint::Ptr mp = Memory::GetMapPoint( backup_pair.first );
        if ( mp->_extra_obs.size() < 5 ) 
            continue; 
        double update = (mp->_pos_world - backup_pair.second).norm();
        LOG(INFO) << "update = "<<update;
        if ( update < 0.01 ) 
            mp->_converged = true; 
        if ( update > 20 ) {
            LOG(WARNING)<< "map point "<<mp->_id <<" changed from "<<backup_pair.second.transpose()<<" to "<<
                mp->_pos_world.transpose()<<endl;
        }
    }
}
 

bool LocalMapping::TestDirectMatch ( 
    Frame::Ptr current, const MatchPointCandidate& candidate,
    Vector2d& px_curr
)
{
    Frame::Ptr ref = Memory::GetFrame(candidate._observed_keyframe);
    MapPoint::Ptr map_point = Memory::GetMapPoint( candidate._map_point );
    
    /*
#ifdef DEBUG_VIZ
    // show the reference and current 
    cv::Mat ref_img = ref->_color.clone();
    cv::Point2f px_ref( candidate._keyframe_pixel[0], candidate._keyframe_pixel[1] );
    cv::rectangle( ref_img, px_ref+cv::Point2f(-6,-6), px_ref+cv::Point2f(6,6), cv::Scalar(0,250,0),3 );
    
    cv::imshow( "ref img", ref_img );
    
    // and the current 
    cv::Point2f px_curr_tmp ( candidate._projected_pixel[0], candidate._projected_pixel[1] );
    cv::Mat curr_img = current->_color.clone();
    cv::rectangle( curr_img, px_curr_tmp+cv::Point2f(-6,-6), px_curr_tmp+cv::Point2f(6,6), cv::Scalar(0,250,0),3 );
    
    cv::imshow( "curr img", curr_img );
    cv::waitKey(0);
#endif
    */
    
    // affine warp 
    Eigen::Matrix2d A_c_r;
    // 计算ref到current的仿射变换 
    utils::GetWarpAffineMatrix( 
        ref, 
        current, 
        candidate._keyframe_pixel,
        map_point->GetObservedPt( candidate._observed_keyframe ),
        map_point->_pyramid_level,
        current->_T_c_w*ref->_T_c_w.inverse(),
        A_c_r
    );
    
    //LOG(INFO) << "A_c_r = "<< A_c_r << endl;
    
    // 应用affine warp 
    int search_level = utils::GetBestSearchLevel( A_c_r, _pyramid_level-1 );
    // int search_level = 0;
    //LOG(INFO) << "search level= "<<search_level<<endl;
    
    WarpAffine( 
        A_c_r, 
        ref->_pyramid[ map_point->_pyramid_level ],
        candidate._keyframe_pixel,
        map_point->_pyramid_level,
        search_level,
        WarpHalfPatchSize+1,
        _patch_with_border
    );
    
    
    // 去掉边界
    uint8_t* ref_patch_ptr = _patch;
    for ( int y=1; y<WarpPatchSize+1; ++y, ref_patch_ptr += WarpPatchSize )
    {
        uint8_t* ref_patch_border_ptr = _patch_with_border + y* ( WarpPatchSize+2 ) + 1;
        for ( int x=0; x<WarpPatchSize; ++x )
            ref_patch_ptr[x] = ref_patch_border_ptr[x];
    }
    
    Vector2d px_scaled ( candidate._projected_pixel/(1<<search_level));
    
    bool success = false;
    
    // align the current pixel 
    success = utils::Align2D(
        current->_pyramid[search_level], 
        _patch_with_border,
        _patch,
        15,
        px_scaled
    );
    
    px_curr = px_scaled*(1<<search_level);
    
    /*
#ifdef DEBUG_VIZ
    // show the original and warpped patch 
    cv::Rect2d rect( 
        px_scaled[0] -WarpHalfPatchSize, 
        px_scaled[1] -WarpHalfPatchSize,
        WarpPatchSize,
        WarpPatchSize
    );
    
    cv::Rect2d rect_ref( 
        candidate._keyframe_pixel[0] -WarpHalfPatchSize, 
        candidate._keyframe_pixel[1] -WarpHalfPatchSize,
        WarpPatchSize,
        WarpPatchSize
    );
    
    // and the affine warpped patch 
    cv::Mat ref_patch( WarpPatchSize+2, WarpPatchSize+2, CV_8UC1 );
    for ( size_t i=0; i<WarpPatchSize+2; i++ )
        for ( size_t j=0; j<WarpPatchSize+2; j++ )
        {
            ref_patch.ptr<uchar>(i)[j] = _patch_with_border[ i*(WarpPatchSize+2) + j];
        }
        
    cv::Mat ref_original = Memory::GetFrame(candidate._observed_keyframe)->_pyramid[0]( rect_ref ).clone();
    cv::namedWindow("ref patch", CV_WINDOW_NORMAL);
    cv::imshow( "ref patch", ref_original );
    cv::resizeWindow("ref patch", 500, 500);
    
    cv::namedWindow("ref affined patch", CV_WINDOW_NORMAL);
    cv::imshow( "ref affined patch", ref_patch );
    cv::resizeWindow("ref affined patch", 500, 500);
    
    cv::Mat matched_patch = current->_pyramid[search_level](rect).clone();
    cv::namedWindow("curr patch", CV_WINDOW_NORMAL);
    cv::imshow( "curr patch", matched_patch );
    cv::resizeWindow("curr patch", 500, 500);
    
    cv::waitKey(1);
    curr_img = current->_color.clone();
    px_curr_tmp = cv::Point2f( px_curr[0], px_curr[1] );
    cv::rectangle( curr_img, px_curr_tmp+cv::Point2f(-6,-6), px_curr_tmp+cv::Point2f(6,6), cv::Scalar(0,250,0),3 );
    cv::imshow( "curr img", curr_img );
    cv::waitKey(0);
#endif
    */
    
    // LOG(INFO) << "px curr change from "<<candidate._projected_pixel.transpose()<<" to "<< px_curr.transpose()<<endl;
    
    return success;
}




    
}