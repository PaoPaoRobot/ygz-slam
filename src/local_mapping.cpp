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
    // 建立匹配点的网格
    
    LOG(INFO) << "frames in local map: "<< this->_local_keyframes.size()<< endl;
    LOG(INFO) << "map points in local map: "<< this->_local_map_points.size() << endl;
    
    _grid.clear();
    _grid.resize( _grid_rows*_grid_cols );
    
    int cntCandidate =0;
    for ( auto it = _local_map_points.begin(); it!=_local_map_points.end(); it++ ) {
        MapPoint::Ptr map_point = Memory::GetMapPoint( *it );
        
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
    // 寻找每个网格中，和当前帧的匹配
    
    int cntSuccess = 0;
    int cntFailed = 0;
    
    set<unsigned long> matched_points; 
    for ( size_t i=0; i<_grid.size(); i++ ) {
        for ( size_t j=0; j<_grid[i].size(); j++ ) {
            // 类似于光流的匹配 
            Vector2d matched_px(0,0); 
            MatchPointCandidate candidate = _grid[i][j];
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
            
            if ( matched_points.find(candidate._map_point) == matched_points.end() ) { // 这个地图点没有被匹配过
                // 在current frame里增加一个对该地图点的观测，以便将来使用
                current->_map_point.push_back( candidate._map_point );
                // 匹配到的点作为观测值，深度值未知，置1
                current->_observations.push_back( Vector3d(matched_px[0], matched_px[1], 1) );
                matched_points.insert( candidate._map_point );
                cntSuccess++;
                break; // 如果这个grid里有成功的匹配，就去掉剩余的点
            }
        }
    } 
    
    LOG(INFO) << "success = " << cntSuccess << ", failed = "<<cntFailed << endl; 
    LOG(INFO) << "matched pixels in current frame: " << current->_map_point.size()<< endl;
    
    if ( current->_map_point.size() < 10 ) { // TODO magic number, adjust it!
        // 匹配不够
        LOG(WARNING) << "insufficient matched pixels, abort this frame ... "<< endl;
        return false;
    }
        
    // Step 3
    // 优化当前帧的pose，类似pnp的做法
    opti::OptimizePoseCeres( current );
    // remove the outliers by reprojection 
    auto iter_obs = current->_observations.begin();
    auto iter_pt = current->_map_point.begin();
    
    int cnt_outlier = 0;
    for ( ; iter_obs!=current->_observations.end();  ) {
        MapPoint::Ptr map_point = Memory::GetMapPoint( *iter_pt );
        Vector2d px = current->_camera->World2Pixel( map_point->_pos_world, current->_T_c_w );
        Vector2d obs = iter_obs->head<2>();
        double error = (px-obs).norm();
        LOG(INFO) << "reprojection error = "<< error << endl;
        
        if ( error > 5 ) { // magic number again! 
            // 重投影误差太大，认为这是个外点
            // TODO 想清楚这里是直接去掉呢，还是按照重投影来计算？
            // 如果计算重投影，那么由于该点被遮挡，它的纹理可能和map point处的纹理非常不同，这会使得local mapping在匹配模板时失败
            // 但是，直接去掉的话，又可能导致当前帧观测的数量太少，后一帧跟不上？
            
            // reset 也有点问题，可能把本该看不见的东西变成可见了？
            // 由于被遮挡，该点的深度就会发生改变，导致和其他特征点出现不一致
            
            // iter_obs = current->_observations.erase( iter_obs );
            // iter_pt = current->_map_point.erase( iter_pt );
            (*iter_obs) [2] = -1;
            iter_obs++;
            iter_pt++;
            
            cnt_outlier++;
            continue; 
        } else {
            iter_obs++;
            iter_pt++;
        }
    }
    
    LOG(INFO) << "outliers = " << cnt_outlier << endl;
    // 现在当前帧应该全是内点了吧，再优化一次以求更精确
    opti::OptimizePoseCeres( current );
    
    // 把地图点按照重投影位置设置一下
    iter_obs = current->_observations.begin();
    iter_pt = current->_map_point.begin();
    
    for ( ; iter_obs!=current->_observations.end(); iter_obs++, iter_pt++ ) {
        if ( (*iter_obs)[2] > 0 ) // inlier
            continue;
        // reset the outlier's pose 
        MapPoint::Ptr map_point = Memory::GetMapPoint( *iter_pt );
        Vector2d px = current->_camera->World2Pixel( map_point->_pos_world, current->_T_c_w );
        iter_obs->head<2>() = px;
        (*iter_obs)[2] = 1;
    }
    return true;
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
    
    // LOG(INFO) << "A_c_r = "<< A_c_r << endl;
    
    // 应用affine warp 
    int search_level = utils::GetBestSearchLevel( A_c_r, _pyramid_level );
    // LOG(INFO) << "search level= "<<search_level<<endl;
    
    WarpAffine( 
        A_c_r, 
        ref->_pyramid[ map_point->_pyramid_level ],
        candidate._keyframe_pixel,
        map_point->_pyramid_level,
        search_level,
        WarpHalfPatchSize+1,
        _patch_with_border
    );
    
    /*
#ifdef DEBUG_VIZ
    // show the original and warpped patch 
    cv::Rect2d rect( 
        candidate._keyframe_pixel[0] * (1<<map_point->_pyramid_level) -WarpHalfPatchSize, 
        candidate._keyframe_pixel[1] * (1<<map_point->_pyramid_level) -WarpHalfPatchSize,
        WarpPatchSize,
        WarpPatchSize
    );
    
    cv::namedWindow("ref patch", CV_WINDOW_NORMAL);
    cv::Mat ref_patch = ref->_pyramid[map_point->_pyramid_level](rect);
    cv::imshow( "ref patch", ref_patch );
    cv::resizeWindow("ref patch", 500, 500);
    
    // and the affine warpped patch 
    cv::Mat curr_patch( WarpPatchSize+2, WarpPatchSize+2, CV_8UC1 );
    for ( size_t i=0; i<WarpPatchSize+2; i++ )
        for ( size_t j=0; j<WarpPatchSize+2; j++ )
        {
            curr_patch.ptr<uchar>(i)[j] = _patch_with_border[ i*(WarpPatchSize+2) + j];
        }
    
    cv::namedWindow("curr patch", CV_WINDOW_NORMAL);
    cv::imshow( "curr patch", curr_patch );
    cv::resizeWindow("curr patch", 500, 500);
    
    cv::waitKey(0);
#endif
    */
    
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
        10,
        px_scaled
    );
    
    px_curr = px_scaled*(1<<search_level);
    
    /*
#ifdef DEBUG_VIZ
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