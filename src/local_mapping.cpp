#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "ygz/local_mapping.h"
#include "ygz/map_point.h"
#include "ygz/frame.h"
#include "ygz/memory.h"
#include "ygz/utils.h"
#include "ygz/optimizer.h"
#include "ygz/feature_detector.h"
#include "ygz/ORB/ORBExtractor.h"
#include "ygz/ORB/ORBMatcher.h"

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
    
    _detector = new FeatureDetector();
    _orb_extractor = new ORBExtractor();
}

void LocalMapping::AddKeyFrame ( Frame* keyframe )
{
    assert( keyframe->_is_keyframe==true );
    _new_keyframes.push_back( keyframe );
    
    // _local_keyframes.push_back( keyframe->_id );
    // _local_keyframes.insert( keyframe->_id );
    
    // TODO 考虑把新增的keyframe中的地图点存储起来，并去掉视野外的点
    /*
    for ( auto& obs: keyframe->_obs ) {
        MapPoint* map_point = Memory::GetMapPoint( obs.first );
        if ( map_point->_bad ) continue;
        _local_map_points.insert( obs.first );
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
    */
}

void LocalMapping::AddMapPoint ( const long unsigned int& map_point_id )
{
    assert( Memory::GetMapPoint(map_point_id) != nullptr );
    _local_map_points.insert( Memory::GetMapPoint(map_point_id) );
}

// 寻找地图与当前帧之间的匹配，当前帧需要有位姿的粗略估计
// 这一步有点像光流
// 希望得到的匹配点能在图像中均匀分布，而不要扎堆在一起，所以使用网格进行区分
// 如果匹配数量足够，调用一次仅有pose的优化
bool LocalMapping::TrackLocalMap ( Frame* current )
{
    // Step 1
    // 建立匹配点的网格，提候选点
    LOG(INFO) << "frames in local map: "<< this->_local_keyframes.size()<< endl;
    LOG(INFO) << "map points in local map: "<< this->_local_map_points.size() << endl;
    
    _grid.clear();
    _grid.resize( _grid_rows*_grid_cols );
    
    // 当前帧根据 sparse alignment 的结果呢，自带一些observation，它们肯定在视野中
    // 但是 alignement 只优化 pose，observation是重投影来的，不一定对
    
    int cntCandidate =0;
    for ( auto& obs_pair: current->_obs ) {
        MapPoint* mp = Memory::GetMapPoint( obs_pair.first );
        // 检测该顶点的 observation 里，有没有在相邻关键帧的
        // 该点已经匹配过了，所以不要再投影一遍
        mp->_track_in_view = true; 
        Vector2d px_curr = obs_pair.second.head<2>(); // 当前帧中的投影（按pose计算的重投影）
        
        // 遍历该点的观测，看哪些观测存在于 local keyframe 中
        // 位于local keyframes 中的才参与比较
        for ( auto& obs_pair: mp->_obs ) {
            Frame* frame = Memory::GetFrame( obs_pair.first );
            if ( _local_keyframes.find( frame )!=_local_keyframes.end() ) {
                MatchPointCandidate candidate;
                candidate._observed_keyframe = obs_pair.first;
                candidate._map_point = mp->_id;
                candidate._keyframe_pixel = obs_pair.second.head<2>();
                candidate._projected_pixel = px_curr;
                int k = static_cast<int> ( px_curr[1]/_cell_size ) *_grid_cols
                            + static_cast<int> ( px_curr[0]/_cell_size );
                _grid[k].push_back( candidate );
                cntCandidate++;
            }
        }
    }
    
    // 遍历local map points，寻找在当前帧视野范围内，而且能匹配上patch的那些点
    for ( auto it = _local_map_points.begin(); it!=_local_map_points.end(); it++ ) {
        
        MapPoint* map_point = *it;
        if ( map_point->_bad == true )
            continue;
        if ( map_point->_track_in_view == true ) // 已经在视野里了
            continue;
        
        // 检查这个地图点是否可见
        Vector3d pt_curr = current->_camera->World2Camera( map_point->_pos_world, current->_T_c_w );
        Vector2d px_curr = current->_camera->Camera2Pixel( pt_curr );
        if ( pt_curr[2] < 0 || !current->InFrame(px_curr) ) { // 在相机后面或不在视野内 
            map_point->_track_in_view = false;
            continue;
        }
        
        // 检测该顶点的 observation 里，有没有在相邻关键帧的
        for ( auto& obs_pair: map_point->_obs ) {
            Frame* frame = Memory::GetFrame( obs_pair.first );
            if ( _local_keyframes.find( frame )!=_local_keyframes.end() ) {
                MatchPointCandidate candidate;
                
                candidate._observed_keyframe = obs_pair.first;
                candidate._map_point = map_point->_id;
                candidate._keyframe_pixel = obs_pair.second.head<2>();
                candidate._projected_pixel = px_curr;
                
                int k = static_cast<int> ( px_curr[1]/_cell_size ) *_grid_cols
                            + static_cast<int> ( px_curr[0]/_cell_size );
                            
                _grid[k].push_back( candidate );
                cntCandidate++;
            }
        }
    }
    
    LOG(INFO) << "Find total "<<cntCandidate <<" candidates."<<endl;
    
    // Step 2 
    // 计算每个候选点和当前帧能否匹配上
    // 要使用到候选点的参考图像以及当前帧的图像
    
    int cntSuccess = 0;
    int cntFailed = 0;
    
    set<unsigned long> matched_points; 
    for ( size_t i=0; i<_grid.size(); i++ ) {
        for ( size_t j=0; j<_grid[i].size(); j++ ) {
            
            // 类似于光流的匹配 
            MatchPointCandidate candidate = _grid[i][j];
            Vector2d matched_px = candidate._projected_pixel; 
            Vector2d projected_px = matched_px;
            bool success = TestDirectMatch( current, candidate, matched_px );
            if ( !success ) {  // 没有匹配到
                // 我要看一下怎么样的匹配才算失败
                Mat img_ref = Memory::GetFrame(candidate._observed_keyframe)->_color.clone();
                Mat img_curr = current->_color.clone();
                cv::circle( img_ref, cv::Point2f( candidate._keyframe_pixel[0], candidate._keyframe_pixel[1]), 2, cv::Scalar(0,0,250), 2);
                cv::circle( img_curr, cv::Point2f( matched_px[0], matched_px[1]), 2, cv::Scalar(0,0,250), 2);
                cv::circle( img_curr, cv::Point2f( projected_px[0], projected_px[1]), 2, cv::Scalar(250,0,250), 2);
                cv::imshow("wrong match ref", img_ref );
                cv::imshow("wrong match curr", img_curr );
                cv::waitKey(0);
                cntFailed++;
                continue;
            } 
            
            Mat img_ref = Memory::GetFrame(candidate._observed_keyframe)->_color.clone();
            Mat img_curr = current->_color.clone();
            // 绿的是优化值，紫红的是初始值
            cv::circle( img_ref, cv::Point2f( candidate._keyframe_pixel[0], candidate._keyframe_pixel[1]), 2, cv::Scalar(0,250,0), 2);
            cv::circle( img_curr, cv::Point2f( matched_px[0], matched_px[1]), 2, cv::Scalar(0,250,0), 2);
            cv::circle( img_curr, cv::Point2f( projected_px[0], projected_px[1]), 2, cv::Scalar(250,0,250), 2);
            cv::imshow("correct match ref", img_ref );
            cv::imshow("correct match curr", img_curr );
            cv::waitKey(0);
            
            if ( matched_points.find(candidate._map_point) == matched_points.end() ) { // 这个地图点没有被匹配过
                // 在current frame里增加一个对该地图点的观测，以便将来使用
                current->_obs[candidate._map_point] = Vector3d( matched_px[0], matched_px[1], 1 );
                // 匹配到的点作为观测值
                matched_points.insert( candidate._map_point );
                cntSuccess++;
                // break; // 如果这个map point已经被匹配过了，就没必要再匹配了
            } else {
                // 这个地图点已经匹配过了
                // break;
            }
        }
    } 
    
    LOG(INFO) << "success = " << cntSuccess << ", failed = "<<cntFailed << endl; 
    if ( current->_obs.size() < 10 ) { // TODO magic number, adjust it!
        // 匹配不够
        LOG(WARNING) << "insufficient matched pixels, abort this frame ... "<< endl;
        return false;
    }
    
    // 至此，current->_obs中已经记录了正确的地图点匹配信息
        
    // Step 3
    // optimize the current pose and the structure 
    map<unsigned long, bool> outlier;
    opti::OptimizePoseCeres( current, outlier );
    
    // remove the outliers by reprojection 
    int cntInliers = 0;
    for ( auto iter = current->_obs.begin(); iter!=current->_obs.end();) {
        if ( outlier[iter->first] ) {
            iter = current->_obs.erase( iter );
        } else {
            
            MapPoint* mp = Memory::GetMapPoint( iter->first );
            Vector3d pt = current->_camera->World2Camera( mp->_pos_world, current->_T_c_w);
            iter->second[2] = pt[2];
            
            /*
            Vector3d pt_from_px = current->_camera->Pixel2Camera( iter->second.head<2>(), pt[2]);
            mp->_extra_obs.push_back( ExtraObservation(current->_T_c_w, pt_from_px) );
            */
            
            // 更新地图点的统计量
            mp->_cnt_visible++;
            cntInliers++;
            iter++;
        }
    }
    
    LOG(INFO) << "inliers: "<<cntInliers<<endl;
    if ( cntInliers < _options.min_track_localmap_inliers ) 
        return false;
    
    // Step 4
    // 更新地图点的统计量
    opti::OptimizeMapPointsCeres( current );
    
    // 把观测按照重投影位置设置一下
    // 这里设重投影位置待议，由于误差存在，可能重投影位置不对，而之前至少是模板匹配得来的
    
    // 看看重投影和obs有何不同
    cv::Mat img_show = current->_color.clone();
    for ( auto iter=current->_obs.begin(); iter!=current->_obs.end(); iter++ ) {
        // reset the observation 
        MapPoint* map_point = Memory::GetMapPoint( iter->first );
        Vector3d pt = current->_camera->World2Camera( map_point->_pos_world, current->_T_c_w );
        Vector2d px = current->_camera->World2Pixel( map_point->_pos_world, current->_T_c_w );
        
        Vector3d pt_obs = current->_camera->Pixel2Camera( px );
        
        cv::circle( img_show, cv::Point2f( (iter->second)[0], (iter->second)[1] ), 2, cv::Scalar(0,0,250), 2 );
        cv::circle( img_show, cv::Point2f( px[0], px[1] ), 2, cv::Scalar(0,250,0), 2 );
        
        iter->second.head<2>() = px;
        
        // 在地图点中添加额外观测
        // if ( map_point->_converged == false )
            // map_point->_extra_obs.push_back( ExtraObservation( pt_obs, current->_T_c_w) );
        iter->second[2] = pt[2]; // 只重设一下距离试试？
    }
    cv::imshow("obs vs reproj" , img_show);
    cv::waitKey(0);
    
    LOG(INFO) << "final observations: "<<current->_obs.size()<<endl;
    
    return true;
}

void LocalMapping::LocalBA ( Frame* current )
{
    // 待优化的关键帧
    set<Frame*> local_keyframes;
    local_keyframes.insert( current );
    vector<Frame*> neighbours = current->GetBestCovisibilityKeyframes();
    for ( Frame* n: neighbours ) 
        local_keyframes.insert( n );
    
    // 待优化的地图点，也就是所有相邻关键帧的观测点
    set<MapPoint*> local_points; 
    for ( Frame* k: local_keyframes ) {
        for ( auto& obs: k->_obs ) {
            MapPoint* mp = Memory::GetMapPoint( obs.first );
            if ( local_points.find(mp) == local_points.end() ) 
                local_points.insert( mp );
        }
    }
    
    LOG(INFO) << "local keyframes: "<< local_keyframes.size()<<endl;
    LOG(INFO) << "local points: "<< local_points.size()<<endl;
    
    // 但是这些个 local points 还可能被其他帧观测到
    /*
    set<Frame*> extra_frames;
    for ( MapPoint* mp: local_points ) {
        for ( auto& obs: mp->_obs ) {
            Frame* f = Memory::GetFrame( obs.first );
            if ( local_keyframes.find(f) == local_keyframes.end() )  {
                // 虽然有观测但不在局部关键帧中
                extra_frames.insert(f);
            }
        }
    }
    */
    
    // setup ceres  
    map<unsigned long, Vector6d*> poses;  
    for( Frame* f: local_keyframes ) {
        Vector6d* pose = new Vector6d();
        Vector3d r = f->_T_c_w.so3().log(), t=f->_T_c_w.translation();
        pose->head<3>() = t;
        pose->tail<3>() = r;
        poses[f->_id] = pose;
    }
    
    ceres::Problem problem;
    
    // TODO 考虑 ordering 和 稀疏性加速
    
    map<unsigned long, Vector3d, less<unsigned long>, Eigen::aligned_allocator<Vector3d>> mp_backup; // 缓存优化前的地图点坐标
    
    // 遍历所有地图点
    for ( MapPoint* mappoint: local_points ) {
        // 缓存之 
        mp_backup[mappoint->_id] = mappoint->_pos_world;
        
        for ( auto& obs: mappoint->_obs ) {
            Frame* f = Memory::GetFrame( obs.first );
            
            Vector3d pt = f->_camera->Pixel2Camera( obs.second.head<2>() );
            
            if ( local_keyframes.find(f) != local_keyframes.end() ) {
                // 该点被局部关键帧观测到，既优化 pose 也优化 point 
                if ( f->_id == 0 ) { // 初始帧不动 
                    problem.AddResidualBlock (
                        new ceres::AutoDiffCostFunction<CeresReprojectionErrorPointOnly,2,3> (
                            new CeresReprojectionErrorPointOnly ( pt.head<2>(), f->_T_c_w )
                        ),
                        new ceres::HuberLoss(5),
                        mappoint->_pos_world.data()
                    );
                } else {
                    // 其他帧参与优化
                    problem.AddResidualBlock (
                        new ceres::AutoDiffCostFunction<CeresReprojectionError,2,6,3> (
                            new CeresReprojectionError ( pt.head<2>() )
                        ),
                        new ceres::HuberLoss(5),
                        poses[f->_id]->data(),
                        mappoint->_pos_world.data()
                    );
                }
            } else {
                // 该点被其他关键帧观测到，此时只优化 point 而不用优化 pose 
                // 这好像被称为 marg ?
                problem.AddResidualBlock (
                    new ceres::AutoDiffCostFunction<CeresReprojectionErrorPointOnly,2,3> (
                        new CeresReprojectionErrorPointOnly ( pt.head<2>(), f->_T_c_w )
                    ),
                    new ceres::HuberLoss(5),
                    mappoint->_pos_world.data()
                );
            }
        }
    }
    
    /*
    for ( auto iter = current->_obs.begin(); iter != current->_obs.end(); iter++ ) {
        MapPoint* mp = Memory::GetMapPoint( iter->first );
        Vector3d pt_curr = current->_camera->Pixel2Camera( iter->second.head<2>() );
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
                Frame* obs_frame = Memory::GetFrame( obs_pair.first );
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
    */
    
    // Optimize it! 
    // 解之
    ceres::Solver::Options options;
    // options.linear_solver_type = ceres::DENSE_SCHUR;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve ( options, &problem, &summary );
    cout<< summary.FullReport() << endl;
    
    // TODO setup inliers and outliers 
    
    
    for ( MapPoint* mappoint: local_points ) {
        for ( auto iter = mappoint->_obs.begin(); iter!=mappoint->_obs.end(); ) {
            // 计算重投影误差
            Frame* f = Memory::GetFrame(iter->first);
            // LOG(INFO) << "obs = "<<iter->second.head<2>().transpose()<<", predict = "<< f->_camera->World2Pixel( mappoint->_pos_world, f->_T_c_w ).transpose();
            Vector2d delta = iter->second.head<2>() - f->_camera->World2Pixel( mappoint->_pos_world, f->_T_c_w );
            if ( delta.dot(delta) > _options.outlier_th ) {
                // 这个观测不够好，从keyframe和地图点中删除彼此的观测
                LOG(INFO) << "reprojection error too large: "<<delta.norm()<<endl;
                iter = mappoint->_obs.erase( iter );
                auto iter_kf = f->_obs.find( mappoint->_id );
                f->_obs.erase( iter_kf );
            } else  {
                iter++;
            }
        }
    }
    
    // set the poses of keyframes 
    for ( auto& pose:poses ) {
        Frame* f = Memory::GetFrame( pose.first );
        f->_T_c_w = SE3( SO3::exp(pose.second->tail<3>()), pose.second->head<3>() );
        delete pose.second;  // new 出来的，记得delete
    }
    
    // 判断地图点是否收敛
    for ( auto& backup_pair: mp_backup ) {
        MapPoint* mp = Memory::GetMapPoint( backup_pair.first );
        if ( mp->_extra_obs.size() < 5 ) 
            continue; 
        double update = (mp->_pos_world - backup_pair.second).norm();
        if ( update < 0.01 ) {
            mp->_bad = false;
            mp->_converged = true; 
        }
        if ( update > 20 ) {
            LOG(WARNING)<< "map point "<<mp->_id <<" changed from "<<backup_pair.second.transpose()<<" to "<<
                mp->_pos_world.transpose()<<endl;
            mp->_bad = true; 
        }
    }
}
 

bool LocalMapping::TestDirectMatch ( 
    Frame* current, const MatchPointCandidate& candidate,
    Vector2d& px_curr
)
{
    Frame* ref = Memory::GetFrame(candidate._observed_keyframe);
    MapPoint* map_point = Memory::GetMapPoint( candidate._map_point );
    
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
    /*
    success = utils::Align2D(
        current->_pyramid[search_level], 
        _patch_with_border,
        _patch,
        10,
        px_scaled,
        false
    );
    */
    success = utils::Align2DCeres( current->_pyramid[search_level], _patch, px_scaled );
    
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
    cv::Mat curr_img = current->_color.clone();
    cv::Point2f px_curr_tmp ( px_curr[0], px_curr[1] );
    cv::rectangle( curr_img, px_curr_tmp+cv::Point2f(-6,-6), px_curr_tmp+cv::Point2f(6,6), cv::Scalar(0,250,0),3 );
    cv::imshow( "curr img", curr_img );
    cv::waitKey(0);
#endif
    */
    // LOG(INFO) << "px curr change from "<<candidate._projected_pixel.transpose()<<" to "<< px_curr.transpose()<<endl;
    
    return success;
}

// 更新局部关键帧: _local_keyframes 
// 寻找与当前帧的地图点有较好共视关系的那些关键帧
void LocalMapping::UpdateLocalKeyframes ( Frame* current )
{
    _local_keyframes.clear();
    map<Frame*,int> keyframeCounter;
    for ( auto& obs_pair: current->_obs ) {
        MapPoint* mp = Memory::GetMapPoint( obs_pair.first );
        if ( mp->_bad == true ) 
            continue; 
        for ( auto& obs_map : mp->_obs ) {
            Frame* frame = Memory::GetFrame( obs_map.first );
            keyframeCounter[frame] ++ ;
        }
    }
    
    if ( keyframeCounter.empty() )
        return; 
    int max = 0;
    Frame* kfmax = nullptr;
    
    // 策略1：能观测到当前帧 mappoint 的关键帧作为局部关键帧
    // 计算共视观测最多的帧
    for( auto& keyframe_pair: keyframeCounter ) {
        if ( keyframe_pair.first->IsBad() ) 
            continue;
        if ( keyframe_pair.second > max ) {
            max = keyframe_pair.second;
            kfmax = keyframe_pair.first;
        }
        
        _local_keyframes.insert( keyframe_pair.first );
    }
    
    // 策略2：和上次结果中共视程度很高的关键帧也作为局部关键帧
    for ( Frame* frame: _local_keyframes ) {
        vector<Frame*> neighbour = frame->GetBestCovisibilityKeyframes(10);
        for ( Frame* neighbour_kf: neighbour ) {
            if ( neighbour_kf->IsBad() ) 
                continue;
            _local_keyframes.insert( neighbour_kf );
        }
    }
    
    if ( kfmax ) {
        current->_ref_keyframe = kfmax; 
    }
}

// 更新与当前帧有关的地图点
void LocalMapping::UpdateLocalMapPoints ( Frame* current )
{
    _local_map_points.clear();
    // 将局部关键帧中的地图点添加到局部地图中
    
    for ( Frame* frame: _local_keyframes ) {
        for ( auto& obs_pair: frame->_obs ) {
            MapPoint* mp = Memory::GetMapPoint( obs_pair.first );
            if ( mp->_bad )
                continue;
            _local_map_points.insert( mp );
            mp->_track_in_view = false;
        }
    }
}

void LocalMapping::Run()
{
    // while (1) {
    // 调通之后改成多线程形式
    while ( _new_keyframes.size() != 0 ) {
        // 将关键帧插入 memory 并处理共视关系
        ProcessNewKeyFrame(); 
        
        // 剔除不合格的 map points 
        MapPointCulling();
        
        // 通过相邻帧间的特征匹配新建一些地图点
        CreateNewMapPoints();
        
        LOG(INFO) << "current kf obs: " << _current_kf->_obs.size()<<endl;
        
        // 没有新的关键帧要处理
        if ( _new_keyframes.empty() ) {
            // 再搜索当前关键帧以及相邻关键帧之间的匹配
            SearchInNeighbors(); 
        }
        
        if ( _new_keyframes.empty() ) {
            // Local Bundle Adjustment 
            LocalBA( _current_kf );
            
            LOG(INFO) << "current kf obs: " << _current_kf->_obs.size()<<endl;
            // Keyframe Culling 
            
            // 删掉一些没必要存在的关键帧，减少计算量
            KeyFrameCulling();
            LOG(INFO) << "current kf obs: " << _current_kf->_obs.size()<<endl;
        }
        
        // 把这个帧加到闭环检测队列中
    } 
    // TODO 添加多线程功能
    
    //}
}

void LocalMapping::ProcessNewKeyFrame()
{
    _current_kf = _new_keyframes.front();
    _new_keyframes.pop_front();
    
    // 提取该帧的特征点
    _detector->SetExistingFeatures( _current_kf );
    _detector->Detect( _current_kf, false );
    // 计算特征点描述，因为在关键帧层级上计算所以计算量不大
    _orb_extractor->Compute( _current_kf );
    
    _current_kf->ComputeBoW();
    
    // 更新此关键帧观测到的地图点
    for ( auto& obs_pair: _current_kf->_obs ) {
        MapPoint* mp = Memory::GetMapPoint( obs_pair.first );
        if ( mp == nullptr ) {
            mp = Memory::GetMapPoint( obs_pair.first );
            LOG(ERROR) << "map point "<<obs_pair.first <<" does not exist."<<endl;
            continue;
        }
        if ( mp->_bad )
            continue;
        // 查看此地图点的观测中是否含有此关键帧
        if ( mp->_obs.find(_current_kf->_id) == mp->_obs.end()) {
            // 没有此关键帧的观测数据
            mp->_obs[_current_kf->_id] = obs_pair.second;
            // TODO 更新特征点方向和最佳描述子
        } else {
            // 新的特征点？
            // 但是只在双目和rgbd模式才能单帧生成新的特征点
        }
    }
    
    // 更新关键帧之间的连接关系：covisibility 和 essential 
    // Essential 现在还没实现，因为主要在loop closure里用
    _current_kf->UpdateConnections();
}

void LocalMapping::MapPointCulling()
{
    auto it = _recent_mappoints.begin();
    int th_obs = 2;
    
    while( it!=_recent_mappoints.end() ) {
        if ( (*it)->_bad ) {
            it = _recent_mappoints.erase( it );
        } else if ( (*it)->GetFoundRatio() < 0.25 ) {
            // 观测程度不够
            (*it)->_bad = true; 
            it = _recent_mappoints.erase( it );
        } else if ( _current_kf->_id - (*it)->_first_observed_frame >= 2 && (*it)->_cnt_found <= th_obs ) {
            // 从2个帧前拿到，但观测数量太少
            (*it)->_bad = true; 
            it = _recent_mappoints.erase( it );
        } else if ( _current_kf->_id - (*it)->_first_observed_frame >= 2 ) {
            // 从三个帧前拿到但被没有剔除，认为是较好的点，但不再追踪
            it = _recent_mappoints.erase( it );
        } else {
            it++;
        }
    }
}

void LocalMapping::CreateNewMapPoints()
{
    int nn = 20;
    vector<Frame*> neighbour_kf = _current_kf->GetBestCovisibilityKeyframes(nn);
    ORBMatcher matcher;
    
    Vector3d cam_current = _current_kf->GetCamCenter();
    // Eigen::Matrix3d K_inv = _current_kf->_camera->GetCameraMatrix().inverse();
    
    int cnt_new_mappoints = 0;
    
    // 计算当前关键帧和别的关键帧之间的特征匹配关系
    for ( size_t i=0; i<neighbour_kf.size(); i++ ) {
        
        Frame* f2 = neighbour_kf[i];
        Vector3d baseline = cam_current - f2->GetCamCenter();
        double mean_depth, min_depth; 
        f2->GetMeanAndMinDepth( mean_depth, min_depth );
        double ratio_baseline_meandepth = baseline.norm()/mean_depth;
        if ( ratio_baseline_meandepth < 0.01 ) // 特别远？
            continue;
        
        // 通过极线去计算匹配关系
        // Fundamental 矩阵
        SE3 T12 = _current_kf->_T_c_w * f2->_T_c_w.inverse(); 
        // F12 = K^{-T} t_12^x R_12 K^{-1} 
        // E12 = t_12^x R_12, epipolar constraint: y1^T t12^x R12 y2 
        Eigen::Matrix3d E12 = SO3::hat( T12.translation() )*T12.rotation_matrix();
        LOG(INFO) << "E12 = "<<E12<<endl;
        
        vector<pair<int, int>> matched_pairs; 
        int matches = matcher.SearchForTriangulation( _current_kf, f2, E12, matched_pairs);
        
        // 对每个匹配点进行三角化
        for ( int im=0; im<matches; im++ ) {
            
            cv::KeyPoint& kp1 = _current_kf->_map_point_candidates[ matched_pairs[im].first ];
            cv::KeyPoint& kp2 = f2->_map_point_candidates[ matched_pairs[im].second ];
            Vector3d pt1 = _current_kf->_camera->Pixel2Camera( Vector2d(kp1.pt.x, kp1.pt.y) );
            Vector3d pt1_world = _current_kf->_camera->Camera2World( pt1, _current_kf->_T_c_w ); 
            Vector3d pt2 = f2->_camera->Pixel2Camera( Vector2d(kp2.pt.x, kp2.pt.y) );
            Vector3d pt2_world = f2->_camera->Camera2World( pt2, f2->_T_c_w ); 
            
            double cos_para_rays = pt1.dot(pt2) / (pt1.norm()*pt2.norm()) ;
            if ( cos_para_rays >= 0.9998 ) // 两条线平行性太强，不好三角化
                continue; 
            double depth1 = 0, depth2=0;
            bool ret = utils::DepthFromTriangulation( T12.inverse(), pt1, pt2, depth1, depth2 ); 
            if ( ret==false || depth1 <0 )
                continue; 
            // 计算三角化点的重投影误差
            Vector3d pt1_triangulated = pt1*depth1;
            Vector2d px2_reproj = f2->_camera->Camera2Pixel( T12.inverse()*pt1_triangulated );
            double reproj_error = (px2_reproj - Vector2d(kp2.pt.x, kp2.pt.y) ).norm();
            if ( reproj_error > 5.991 )  // 重投影太大
                continue; 
            
            // 什么是尺度连续性。。。算了先不管它
            
            // 终于可以生成地图点了
            MapPoint* mp = Memory::CreateMapPoint();
            mp->_first_observed_frame = _current_kf->_id;
            mp->_pyramid_level = kp1.octave;
            mp->_obs[_current_kf->_id] = Vector3d(kp1.pt.x, kp1.pt.y, depth1);
            mp->_obs[ f2->_id ] = Vector3d( kp2.pt.x, kp2.pt.y, depth2 );
            mp->_cnt_visible = 2;
            mp->_cnt_found = 2;
            mp->_pos_world = _current_kf->_camera->Camera2World( pt1*depth1, _current_kf->_T_c_w );
            
            _current_kf->_obs[mp->_id] = mp->_obs[_current_kf->_id];
            f2->_obs[mp->_id] = mp->_obs[f2->_id];
            
            mp->_descriptor.push_back( _current_kf->_descriptors[matched_pairs[im].first] );
            mp->_descriptor.push_back( f2->_descriptors[ matched_pairs[im].second] );
            
            _recent_mappoints.push_back( mp );
            
            _current_kf->_candidate_status[ matched_pairs[im].first ] = CandidateStatus::TRIANGULATED;
            f2->_candidate_status[ matched_pairs[im].second ] = CandidateStatus::TRIANGULATED;
            
            
            cnt_new_mappoints++;
        }
    }
    
    LOG(INFO) << "New map points: " << cnt_new_mappoints << endl;
}

void LocalMapping::SearchInNeighbors()
{
    int nn = 20;
    
    // vector<Frame*> neighbour = _current_kf->GetBestCovisibilityKeyframes(nn);

}

// 剔除一些冗余的关键帧
void LocalMapping::KeyFrameCulling()
{
    // 原则：如果某个关键帧的地图点有90%以上，都能被其他关键帧看到，则认为此关键帧是冗余的
    // 总帧数太少时，不要做culling
    if ( Memory::GetNumberFrames() <= 5 )
        return;
    vector<Frame*> local_keyframes = _current_kf->GetBestCovisibilityKeyframes();
    for ( Frame* frame : local_keyframes ) {
        if ( frame->_id == 0 ) 
            continue;
        
        const int th_obs = 3; 
        int redundant_observations = 0; 
        int mappoints = 0;
        
        for( auto& obs: frame->_obs ) {
            MapPoint* mp = Memory::GetMapPoint( obs.first );
            if ( mp->_bad == false ) {
                int nobs = 0;
                mappoints++;
                if ( mp->_obs.size() > th_obs ) {
                    // 至少被三个帧看到
                    for ( auto& obs_mp : mp->_obs ) {
                        Frame* obs_frame = Memory::GetFrame( obs_mp.first );
                        if ( obs_frame == frame ) 
                            continue;
                        nobs++;
                    }
                }
                
                if ( nobs >= th_obs )
                    redundant_observations++;
            }
        }
        
        if ( redundant_observations > 0.9*mappoints )
            frame->_bad = true; 
    }
    
}


    
}