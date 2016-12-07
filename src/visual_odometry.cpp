#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "ygz/visual_odometry.h"
#include "ygz/optimizer.h"
#include "ygz/memory.h"
#include "ygz/ORB/ORBextractor.h"

namespace ygz {
void VisualOdometry::AddFrame(const Frame::Ptr& frame)
{
    if ( _status == VO_NOT_READY ) {
        _ref_frame = frame;
        _status = VO_INITING;
    }

    _curr_frame = frame;
    _last_status = _status;

    if ( _status == VO_INITING ) {
        // TODO initialization
        MonocularInitialization();
        if ( _status != VO_GOOD )
            return;
    } else {
        bool OK = false;
        // initialized, do tracking
        if ( _status == VO_GOOD ) {
            // 跟踪参考帧，也就是上一个帧
            OK = TrackRefFrame();
            if ( OK ) {
                // 单纯跟踪上一个帧会出现明显的累计误差，所以还要和局部地图进行比对 
                OK = TrackLocalMap();  // compare the current frame with local map 
            }
            
            if ( OK ) {
                _status = VO_GOOD;      // 跟踪成功
            } else {
                _status = VO_LOST;      // 丢失，尝试在下一帧恢复 
            }
            
            
        } else {
            // not good, relocalize
            // TODO: relocalization
        }
    }

    // store the result or other things
}

void VisualOdometry::MonocularInitialization()
{
    if ( _tracker->Status() == Tracker::TRACK_NOT_READY ) {
        // init the tracker
        _tracker->SetReference( _curr_frame );
    } else if ( _tracker->Status() == Tracker::TRACK_GOOD ) {
        // track the current frame
        _tracker->Track( _curr_frame );

        // try intialize, check mean disparity
        if ( !_init.Ready( _tracker->MeanDisparity()) ) {
            // too small disparity, quit
            return;
        }

        // initialization ready, try it
        vector<Vector2d> pt1, pt2;
        vector<Vector3d> pts_triangulated;
        SE3 T12;

        // _tracker->GetTrackedPointsNormalPlane( pt1, pt2 );
        _tracker->GetTrackedPixel( pt1, pt2 );
        bool init_success = _init.TryInitialize( pt1, pt2, _ref_frame, _curr_frame );
        if ( init_success ) {
            // init succeeds, set VO as normal tracking
            
            /** * debug only 
            // check the covisibility 
            for ( unsigned long map_point_id : _ref_frame->_map_point ) {
                MapPoint::Ptr p = Memory::GetMapPoint( map_point_id );
                p->PrintInfo();
            }
            */ 
            
            // two view BA to minimize the reprojection error 
            // use g2o or ceres or what you want 
            // opti::TwoViewBAG2O( _ref_frame->_id, _curr_frame->_id );
            opti::TwoViewBACeres( _ref_frame->_id, _curr_frame->_id );
            // reproject the map points in these two views 
            ReprojectMapPointsInInitializaion();
            
            // set two key-frames and their observed points 
            SetKeyframe( _ref_frame );
            SetKeyframe( _curr_frame );
            
            _status = VO_GOOD;
            _ref_frame = _curr_frame;
            
#ifdef DEBUG_VIZ
            // plot the inliers
            Mat img_show;
            if ( _curr_frame != nullptr) {
                img_show = _curr_frame->_color.clone();
            } else {
                return;
            }

            vector<bool> inliers = _init.GetInliers();
            list<cv::Point2f> pts = _tracker->GetPxCurr();

            // LOG(INFO) << "inlier size=" << inliers.size() <<endl;
            // LOG(INFO) << "points size=" << pts.size() <<endl;
            
            int i=0;
            for ( const cv::Point2f& p: _tracker->GetPxCurr() ) {
                if ( inliers[i] == true )
                    cv::circle( img_show, p, 5, cv::Scalar(0,250,0), 2 );
                else
                    cv::circle( img_show, p, 5, cv::Scalar(0,0,250), 2 ); // red for outliers
                i++;
            }

            // init inliers
            cv::imshow( "inliers of H", img_show );
            cv::waitKey(1);
#endif 
            return;
        } else {
            // init failed, still tracking
        }
    } else {
        // lost, reset the tracker
        LOG(WARNING) << "Tracker has lost, resetting it to initialize " << endl;
        _tracker->SetReference( _curr_frame );
    }
}

void VisualOdometry::SetKeyframe ( Frame::Ptr frame )
{
    frame->_is_keyframe = true; 
    Memory::RegisterFrame( frame, true );
    
    // 在关键帧中，我们把直接法提取的关键点升级为带描述的特征点，以实现全局的匹配
    ORBExtractor orb;
    for ( unsigned long map_point_id: frame->_map_point ) {
        MapPoint::Ptr map_point = Memory::GetMapPoint( map_point_id );
        if ( map_point->_descriptor.data==nullptr ) {
            // 这个地图点还没有描述子，计算它的描述 
            orb.Compute( frame->_pyramid[map_point->_pyramid_level], map_point->_obs[frame->_id].head<2>(), map_point->_keypoint, map_point->_descriptor );
        }
    }
    
    _local_mapping.AddKeyFrame( frame );
}

bool VisualOdometry::TrackRefFrame()
{
    SE3 TCR = _TCR_estimated; 
    for ( int level = _curr_frame->_pyramid.size()-1; level>=0; level -- ) {
        _align[level].SetTCR(TCR);
        _align[level].SparseImageAlignmentCeres( _ref_frame, _curr_frame, level );
        TCR = _align[level].GetEstimatedT21();
    }
    
    // TODO validate the result obtained by sparse image alignment 
    _TCR_estimated = TCR;
    
    // set the pose of current frame 
    _curr_frame->_T_c_w = _TCR_estimated * _ref_frame->_T_c_w;
    
    // 我有点怀疑这里的结果对不对，让我们测试一下
    // 所有地图点在帧0和当前帧的投影
    /*
    Frame::Ptr ref = Memory::GetFrame(0);
    Mat ref_img = Memory::GetFrame(0)->_color.clone();
    Mat curr_img = _curr_frame->_color.clone();
    Mat cr_img = _curr_frame->_color.clone();
    for ( auto map_point_pair : Memory::_points ) {
        MapPoint::Ptr point = map_point_pair.second;
        
        // Vector3d pt_ref = ref->_camera->World2Pixel( point->_pos_world, ref->_T_c_w );
        // Vector3d pt_tcr = _TCR_estimated * pt_ref;
        
        Vector2d px_ref = ref->_camera->World2Pixel( point->_pos_world, ref->_T_c_w);
        Vector2d px_curr = _curr_frame->_camera->World2Pixel( point->_pos_world, _curr_frame->_T_c_w );
        // Vector2d px_tcr = _curr_frame->_camera->Camera2Pixel( pt_tcr );
        
        cv::circle( ref_img, cv::Point2f(px_ref[0], px_ref[1]), 5, cv::Scalar(0,250,0));
        cv::circle( curr_img, cv::Point2f(px_curr[0], px_curr[1]), 5, cv::Scalar(0,250,0));
        // cv::circle( cr_img, cv::Point2f(px_ref[0], px_ref[1]), 5, cv::Scalar(0,250,0));
    }
    cv::imshow("ref frame", ref_img);
    cv::imshow("current frame", curr_img);
    cv::waitKey(0);
    */
    return true; 
}

bool VisualOdometry::TrackLocalMap() 
{
    // Track Local Map 还是有点微妙的
    // 当前帧不是关键帧时，它不会提取新的特征，所以特征点都是从局部地图中投影过来的
    // Step 1. 在局部地图中寻找投影点并匹配之
    list<MatchPointCandidate> candidate = _local_mapping.FindMatchedCandidate( _curr_frame );
    
    // Step 2. 根据这些匹配点优化当前帧的位姿 
    return true;
}

void VisualOdometry::ReprojectMapPointsInInitializaion()
{
    // 遍历地图中的3D点
    for ( auto& map_points_pair: Memory::_points ) {
        MapPoint::Ptr map_point = map_points_pair.second;
        for ( auto& observation: map_point->_obs ) {
            Frame::Ptr frame = Memory::GetFrame(observation.first);
            Vector3d pt = frame->_camera->World2Camera( map_point->_pos_world, frame->_T_c_w );
            Vector2d px = frame->_camera->Camera2Pixel( pt );
            observation.second = Vector3d( px[0], px[1], pt[2]); // set the observed posiiton 
        }
    }
}

    

}
