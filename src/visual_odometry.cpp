#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "ygz/system.h"
#include "ygz/visual_odometry.h"
#include "ygz/optimizer.h"
#include "ygz/memory.h"
#include "ygz/ORB/ORBextractor.h"

namespace ygz {
    
VisualOdometry::VisualOdometry ( System* system )
    : _tracker(new Tracker), _system(system), _depth_filter(new opti::DepthFilter() )
{
    int pyramid = Config::get<int>("frame.pyramid");
    _align.resize( pyramid );
    
    _min_keyframe_rot = Config::get<double>("vo.keyframe.min_rot");
    _min_keyframe_trans = Config::get<double>("vo.keyframe.min_trans");
    _min_keyframe_features = Config::get<int>("vo.keyframe.min_features");
}

VisualOdometry::~VisualOdometry()
{
    if ( _depth_filter )
        delete _depth_filter;
}

    
// 整体接口，逻辑有点复杂，参照orb-slam2设计
bool VisualOdometry::AddFrame(const Frame::Ptr& frame)
{
    if ( _status == VO_NOT_READY ) {
        _ref_frame = frame;
        SetKeyframe(_ref_frame, true);
        _status = VO_INITING;
    }

    _curr_frame = frame;
    _last_status = _status;

    if ( _status == VO_INITING ) {
        // TODO add stereo and RGBD initialization
        MonocularInitialization();
        if ( _status != VO_GOOD )
            return false;
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
                
                // 检查此帧是否可以作为关键帧
                if ( NeedNewKeyFrame() ) {
                    // create new key-frame 
                    LOG(INFO) << "this frame is regarded as a new key-frame. "<< endl;
                    cv::waitKey(0);
                }
                
                // 把参考帧设为当前帧
                _ref_frame = _curr_frame;
                
            } else {
                _status = VO_LOST;      // 丢失，尝试在下一帧恢复 
                return false;
            }
        } else {
            // not good, relocalize
            // TODO: relocalization
        }
    }

    // store the result or other things
    return true;
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
            
            // set current as key-frame and their observed points 
            // SetKeyframe( _ref_frame );
            SetKeyframe( _curr_frame );
            
            _status = VO_GOOD;
            _ref_frame = _curr_frame;
            
#ifdef DEBUG_VIZ
            // 画出初始化点在两个图像中的投影
            Mat img_ref=  _ref_frame->_color.clone();
            Mat img_curr=  _curr_frame->_color.clone();
            
            for ( auto& map_point_pair : Memory::_points ) {
                Vector2d px_ref = _ref_frame->_camera->World2Pixel( map_point_pair.second->_pos_world, _ref_frame->_T_c_w );
                Vector2d px_curr = _curr_frame->_camera->World2Pixel( map_point_pair.second->_pos_world, _curr_frame->_T_c_w );
                
                Vector2d obs_ref = map_point_pair.second->_obs[_ref_frame->_id].head<2>();
                Vector2d obs_curr = map_point_pair.second->_obs[_curr_frame->_id].head<2>();
                
                
                if ( map_point_pair.second->_bad ) {
                    cv::circle( img_ref, cv::Point2f(px_ref[0], px_ref[1]),5, cv::Scalar(0,0,250) );
                    cv::circle( img_curr, cv::Point2f(px_curr[0], px_curr[1]),5, cv::Scalar(0,0,250) );
                    
                    cv::circle( img_ref, cv::Point2f(obs_ref[0], obs_ref[1]),5, cv::Scalar(0,0,250) );
                    cv::circle( img_curr, cv::Point2f(obs_curr[0], obs_curr[1]),5, cv::Scalar(0,0,250) );
                    
                    cv::line( img_ref, cv::Point2f(px_ref[0], px_ref[1]), cv::Point2f(obs_ref[0], obs_ref[1]), cv::Scalar(0,0,250) );
                    cv::line( img_curr, cv::Point2f(px_curr[0], px_curr[1]), cv::Point2f(obs_curr[0], obs_curr[1]), cv::Scalar(0,0,250) );
                    
                } else {
                    cv::circle( img_ref, cv::Point2f(px_ref[0], px_ref[1]),5, cv::Scalar(0,250,0) );
                    cv::circle( img_curr, cv::Point2f(px_curr[0], px_curr[1]),5, cv::Scalar(0,250,0) );
                    
                    cv::circle( img_ref, cv::Point2f(obs_ref[0], obs_ref[1]),5, cv::Scalar(0,250,0) );
                    cv::circle( img_curr, cv::Point2f(obs_curr[0], obs_curr[1]),5, cv::Scalar(0,250,0) );
                    
                    cv::line( img_ref, cv::Point2f(px_ref[0], px_ref[1]), cv::Point2f(obs_ref[0], obs_ref[1]), cv::Scalar(0,250,0), 3);
                    cv::line( img_curr, cv::Point2f(px_curr[0], px_curr[1]), cv::Point2f(obs_curr[0], obs_curr[1]), cv::Scalar(0,250,0), 3 );
                }
                
            }
            
            cv::imshow("ref", img_ref);
            cv::imshow("curr", img_curr);
            cv::waitKey(0);
#endif 
            opti::TwoViewBACeres( _ref_frame->_id, _curr_frame->_id ); // 去掉外点后再优化一次
            
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

void VisualOdometry::SetKeyframe ( Frame::Ptr frame, bool initializing )
{
    frame->_is_keyframe = true; 
    Memory::RegisterFrame( frame, true );
    
    if ( initializing == false ) {
        // 在新的关键帧中，提取新的特征点
        FeatureDetector detector;
        detector.SetExistingFeatures( frame );
        detector.Detect( frame );
    }
    
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
    _last_key_frame = frame; 
    
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
    bool success = _local_mapping.TrackLocalMap( _curr_frame );
    
    // plot the currernt image 
#ifdef DEBUG_VIZ 
    Mat img_show = _curr_frame->_color.clone();
    for ( Vector3d& obs: _curr_frame->_observations ) {
        cv::rectangle( img_show, cv::Point2f( obs[0]-5, obs[1]-5), cv::Point2f( obs[0]+5, obs[1]+5), cv::Scalar(0,250,0), 2 );
        cv::rectangle( img_show, cv::Point2f( obs[0]-1, obs[1]-1), cv::Point2f( obs[0]+1, obs[1]+1), cv::Scalar(0,250,0), 1 );
    }
    cv::imshow("current", img_show);
    cv::waitKey(0);
#endif
    
    return true;
}

void VisualOdometry::ReprojectMapPointsInInitializaion()
{
    // 遍历地图中的3D点
    for ( auto iter = Memory::_points.begin(); iter!=Memory::_points.end(); iter++ ) {
        MapPoint::Ptr map_point = iter->second;
        bool badpoint = false;
        for ( auto& observation: map_point->_obs ) {
            Frame::Ptr frame = Memory::GetFrame(observation.first);
            Vector3d pt = frame->_camera->World2Camera( map_point->_pos_world, frame->_T_c_w );
            Vector2d px = frame->_camera->Camera2Pixel( pt );
            double reproj_error = (px - observation.second.head<2>()).norm();
            // LOG(INFO) << "init reprojection error = "<<reproj_error<<endl;
            if ( reproj_error > _options.init_reproj_error_th ) // 重投影误差超过阈值
            {
                badpoint = true;
                break;
            }
            // observation.second = Vector3d( px[0], px[1], pt[2]); // reset the observed posiiton 
        }
        
        if ( badpoint == true ) {
            // set this point as a bad one, require the memory to remove it 
            map_point->_bad = true; 
        }
    }
}

bool VisualOdometry::NeedNewKeyFrame()
{
    // 天哪orb的keyframe选择策略是怎么想出来的，莫非用机器学习去学的？
    // 话说这事还真能用机器学习学吧...
    // 先简单点，根据上个key-frame和当前的位姿差，以及tracked points做决定
    SE3 deltaT = _last_key_frame->_T_c_w.inverse()*_curr_frame->_T_c_w;
    // 度量旋转李代数和平移向量的范数
    double dRotNorm = deltaT.so3().log().norm();
    double dTransNorm = deltaT.translation().norm();
    
    LOG(INFO) << "rot = "<<dRotNorm << ", t = "<< dTransNorm<<endl;
    bool condition1 = dRotNorm > _min_keyframe_rot || dTransNorm > _min_keyframe_trans;
    
    // 计算正确跟踪的特征数量
    // 如果这个数量太少，很可能前面计算结果也是不对的
    bool condition2 = _curr_frame->_map_point.size() > _min_keyframe_features;
    
    return condition1 && condition2;
}

    

}
