#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "ygz/local_mapping.h"
#include "ygz/memory.h"
#include "ygz/utils.h"

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
        _local_map_points.insert( point_id );
    }
    
    // 删除太远的地图点和 key frame
}

// 寻找地图与当前帧之间的匹配，当前帧需要有位姿的粗略估计
// 这一步有点像光流
// 希望得到的匹配点能在图像中均匀分布，而不要扎堆在一起，所以使用网格进行区分
list< MatchPointCandidate > LocalMapping::FindMatchedCandidate ( Frame::Ptr current )
{
    // Step 1
    // 建立匹配点的网格
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
        if ( !current->InFrame(px_curr, 15) )  // 不在图像中
            continue; 
        // 检测该顶点的 observation 里，有没有在相邻关键帧的
        MatchPointCandidate candidate;
        
        bool exist = false; 
        for ( auto& obs_pair: map_point->_obs ) {
            unsigned long keyframe_id = obs_pair.first;
            for ( auto it=_local_keyframes.begin(); it!=_local_keyframes.end(); it++ ) {
                if ( *it == keyframe_id ) {
                    exist = true; 
                    candidate._observed_keyframe = *it;
                    candidate._map_point =map_point->_id;
                    candidate._keyframe_pixel = obs_pair.second.head<2>();
                    candidate._projected_pixel = px_curr;
                    break;
                }
            }
        }
        
        if ( exist == false ) 
            continue; 
        int k = static_cast<int> ( px_curr[1]/_cell_size ) *_grid_cols
                      + static_cast<int> ( px_curr[0]/_cell_size );
        cntCandidate ++;
        // check if this candidate exist 
        _grid[k].push_back( candidate );
    }
    
    LOG(INFO) << "Find total "<<cntCandidate <<" candidates."<<endl;
    
    // Step 2 
    // 寻找每个网格中，和当前帧的匹配
    for ( size_t i=0; i<_grid.size(); i++ ) {
        for ( size_t j=0; j<_grid[i].size(); j++ ) {
            // 类似于光流的匹配 
            Vector2d matched_px(0,0); 
            bool success = TestDirectMatch( current, _grid[i][j], matched_px );
            if ( !success )  // 没有匹配到
                continue;
            // 在current frame里增加一个对该地图点的观测，以便将来使用
        }
    }
    
    // Step 3
}

bool LocalMapping::TestDirectMatch ( 
    Frame::Ptr current, const MatchPointCandidate& candidate,
    Vector2d& px_curr
)
{
    Frame::Ptr ref = Memory::GetFrame(candidate._observed_keyframe);
    MapPoint::Ptr map_point = Memory::GetMapPoint( candidate._map_point );
    
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
    
    LOG(INFO) << "A_c_r = "<< A_c_r << endl;
    
    // 应用affine warp 
    int search_level = utils::GetBestSearchLevel( A_c_r, _pyramid_level );
    LOG(INFO) << "search level= "<<search_level<<endl;
    
    WarpAffine( 
        A_c_r, 
        ref->_pyramid[ map_point->_pyramid_level ],
        candidate._keyframe_pixel,
        map_point->_pyramid_level,
        search_level,
        WarpHalfPatchSize+1,
        _patch_with_border
    );
    
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
    return success;
}




    
}