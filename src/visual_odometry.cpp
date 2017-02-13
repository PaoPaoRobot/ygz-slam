#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <boost/format.hpp>

#include "ygz/system.h"
#include "ygz/visual_odometry.h"
#include "ygz/optimizer.h"
#include "ygz/memory.h"
#include "ygz/ORB/ORBExtractor.h"
#include "ygz/tracker.h"
#include "ygz/feature_detector.h"
#include "ygz/initializer.h"
#include "ygz/local_mapping.h"
#include "ygz/camera.h"
#include "ygz/ORB/ORBExtractor.h"
#include "ygz/ORB/ORBMatcher.h"

namespace ygz {
    
VisualOdometry::VisualOdometry ( System* system )
    :  _system(system)
{
    int pyramid = Config::get<int>("frame.pyramid");
    _align.resize( pyramid );
    
    _min_keyframe_rot = Config::get<double>("vo.keyframe.min_rot");
    _min_keyframe_trans = Config::get<double>("vo.keyframe.min_trans");
    _min_keyframe_features = Config::get<int>("vo.keyframe.min_features");
    
    // TODO 为了调试方便所以在这里写new，但最后需要挪到system里面管理
    _init = new Initializer();
    _detector = new FeatureDetector();
    _tracker = new Tracker();
    _local_mapping = new LocalMapping();
    _depth_filter = new opti::DepthFilter( _local_mapping );
    
    _orb_extractor = new ORBExtractor();
    _orb_matcher = new ORBMatcher();
}

VisualOdometry::~VisualOdometry()
{
    if ( _depth_filter )
        delete _depth_filter;
    if ( _detector ) 
        delete _detector;
    if ( _tracker )
        delete _tracker;
    if ( _local_mapping )
        delete _local_mapping;
    if ( _init )
        delete _init;
    if ( _orb_extractor ) 
        delete _orb_extractor;
    if ( _orb_matcher ) 
        delete _orb_matcher;
}
    
// 整体接口，逻辑有点复杂，参照orb-slam2设计
bool VisualOdometry::AddFrame( Frame* frame )
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
                LOG(INFO) << "Track success, current TCW = \n" << _curr_frame->_T_c_w.matrix() << endl;
                
                _processed_frames ++; 
                // 检查此帧是否可以作为关键帧
                if ( NeedNewKeyFrame() ) {
                    // create new key-frame 
                    LOG(INFO) << "this frame is regarded as a new key-frame. "<< endl;
                    SetKeyframe( _curr_frame );
                } else {
                    // 该帧是普通帧，加到 depth filter 中去
                    // _depth_filter->AddFrame( frame );
                }
                
                // 把参考帧设为当前帧
                if ( _ref_frame && _ref_frame->_is_keyframe==false )
                    delete _ref_frame;  // 不是关键帧的话就可以删了
                _ref_frame = _curr_frame;
                
                /*
                // plot the currernt image 
                Mat img_show = _curr_frame->_color.clone();
                for ( auto& obs_pair: _curr_frame->_obs ) {
                    Vector3d obs = obs_pair.second;
                    cv::rectangle( img_show, cv::Point2f( obs[0]-5, obs[1]-5), cv::Point2f( obs[0]+5, obs[1]+5), cv::Scalar(0,250,0), 2 );
                    cv::rectangle( img_show, cv::Point2f( obs[0]-1, obs[1]-1), cv::Point2f( obs[0]+1, obs[1]+1), cv::Scalar(0,250,0), 1 );
                }
                cv::imshow("current", img_show);
                cv::waitKey(1);
                */
                
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
        if ( !_init->Ready( _tracker->MeanDisparity()) ) {
            // too small disparity, quit
            return;
        }

        // initialization ready, try it
        vector<Vector2d> pt1, pt2;
        vector<Vector3d> pts_triangulated;
        SE3 T12;

        // _tracker->GetTrackedPointsNormalPlane( pt1, pt2 );
        _tracker->GetTrackedPixel( pt1, pt2 );
        LOG(INFO) <<pt1.size()<<","<<pt2.size()<<endl;
        
        bool init_success = _init->TryInitialize( pt1, pt2, _ref_frame, _curr_frame );
        if ( init_success ) {
            // 初始化算法认为初始化通过，但可能会有outlier和误跟踪
            // 光流在跟踪之后不能保证仍是角点
            // _tracker->PlotTrackedPoints(); 
            
            // 把跟踪后的特征点设为 current 的特征点，计算描述量
            int i = 0;
            
            for ( size_t iRef = 0; iRef<_ref_frame->_map_point_candidates.size(); iRef++ ) {
                if ( _ref_frame->_candidate_status[iRef] == CandidateStatus::BAD ) 
                    continue;
                _curr_frame->_map_point_candidates.push_back(
                    cv::KeyPoint( pt2[i][0], pt2[i][1], _ref_frame->_map_point_candidates[iRef].size )
                );
                _curr_frame->_candidate_status.push_back( CandidateStatus::WAIT_DESCRIPTOR );
                i++;
            }
            
            _curr_frame = Memory::RegisterFrame( _curr_frame, false );
            
            // 计算 ref 和 current 之间的描述子，依照描述子检测匹配是否正确
            CheckInitializationByDescriptors();
            
            // two view BA to minimize the reprojection error 
            // use g2o or ceres or what you want 
            // opti::TwoViewBAG2O( _ref_frame->_id, _curr_frame->_id );
            
            opti::TwoViewBACeres( _ref_frame->_id, _curr_frame->_id, true );
            LOG(INFO) << "total map points: " << Memory::GetAllPoints().size() << endl;
            
            // reproject the map points in these two views 
            // 这步会删掉深度值不对的那些点
            ReprojectMapPointsInInitializaion();
            
            // 去掉外点后再优化一次
            opti::TwoViewBACeres( _ref_frame->_id, _curr_frame->_id, false ); 
            
            // 重置观测以及map scale 
            ResetCurrentObservation();
            
            LOG(INFO) << "ref pose = \n" <<_ref_frame->_T_c_w.matrix()<<endl;
            LOG(INFO) << "current pose = \n"<< _curr_frame->_T_c_w.matrix()<<endl;
            LOG(INFO) << "current obs: " << _curr_frame->_obs.size() << endl;
            
            // set current as key-frame and their observed points 
            SetKeyframe( _curr_frame, true );
            _status = VO_GOOD;
            
            LOG(INFO) << "tracked points: " << pt1.size() << endl;
            LOG(INFO) << "obs: " << _curr_frame->_obs.size() << endl;
            LOG(INFO) << "total map points: " << Memory::GetAllPoints().size() << endl;
            
            /*
#ifdef DEBUG_VIZ
            int cntBad = 0;
            for ( auto& map_point_pair : Memory::_points ) {
                
                if ( map_point_pair.second->_obs.size() < 2 ) // 其中一帧的观测被删
                    continue;
                
                Vector2d px_ref = _ref_frame->_camera->World2Pixel( map_point_pair.second->_pos_world, _ref_frame->_T_c_w );
                Vector2d px_curr = _curr_frame->_camera->World2Pixel( map_point_pair.second->_pos_world, _curr_frame->_T_c_w );
                
                Vector2d obs_ref = map_point_pair.second->_obs[_ref_frame->_id].head<2>();
                Vector2d obs_curr = map_point_pair.second->_obs[_curr_frame->_id].head<2>();
                
                
                if ( map_point_pair.second->_bad ) {
                    cntBad++;
                    cv::circle( img_ref, cv::Point2f(px_ref[0], px_ref[1]),2, cv::Scalar(0,0,250),2 );
                    cv::circle( img_curr, cv::Point2f(px_curr[0], px_curr[1]),2, cv::Scalar(0,0,250),2 );
                    
                    cv::circle( img_ref, cv::Point2f(obs_ref[0], obs_ref[1]),2, cv::Scalar(0,0,250),2 );
                    cv::circle( img_curr, cv::Point2f(obs_curr[0], obs_curr[1]),2, cv::Scalar(0,0,250),2 );
                    
                    cv::line( img_ref, cv::Point2f(px_ref[0], px_ref[1]), cv::Point2f(obs_ref[0], obs_ref[1]), cv::Scalar(0,0,250) );
                    cv::line( img_curr, cv::Point2f(px_curr[0], px_curr[1]), cv::Point2f(obs_curr[0], obs_curr[1]), cv::Scalar(0,0,250) );
                } else { 
                    cv::circle( img_ref, cv::Point2f(px_ref[0], px_ref[1]),2, cv::Scalar(0,250,0),2 );
                    cv::circle( img_curr, cv::Point2f(px_curr[0], px_curr[1]),2, cv::Scalar(0,250,0),2 );
                    
                    cv::circle( img_ref, cv::Point2f(obs_ref[0], obs_ref[1]),2, cv::Scalar(0,250,0),2 );
                    cv::circle( img_curr, cv::Point2f(obs_curr[0], obs_curr[1]),2, cv::Scalar(0,250,0),2 );
                    
                    cv::line( img_ref, cv::Point2f(px_ref[0], px_ref[1]), cv::Point2f(obs_ref[0], obs_ref[1]), cv::Scalar(0,250,0), 3);
                    cv::line( img_curr, cv::Point2f(px_curr[0], px_curr[1]), cv::Point2f(obs_curr[0], obs_curr[1]), cv::Scalar(0,250,0), 3 );
                }
            }
            LOG(INFO) << "bad map points: " << cntBad << endl;
            
            
            cv::imshow("ref", img_ref);
            cv::imshow("curr", img_curr);
            cv::waitKey(0);
#endif 
            */
            _ref_frame = _curr_frame;
            LOG(INFO) <<_ref_frame->_id<<","<<_curr_frame->_id<<endl;
            
            return;
        } else {
            // init failed, still tracking
            delete _curr_frame;
        }
    } else {
        // lost, reset the tracker
        LOG(WARNING) << "Tracker has lost, resetting it to initialize " << endl;
        _tracker->SetReference( _curr_frame );
    }
}

void VisualOdometry::SetKeyframe ( Frame* frame, bool initializing )
{
    frame->_is_keyframe = true; 
    // Memory::PrintInfo();
    if ( frame->_id != 0 ) { // 已经注册了，就不要重复注册一遍
    } else {
        frame = Memory::RegisterFrame( frame );  // 未注册，则注册新的关键帧
    }
    LOG(INFO) << "frame id = " << frame->_id << endl;
    
    if ( initializing == false ) {
        // 检查 observation 中的描述是否匹配
        CheckObservationByDescriptors();
    }
    
    _local_mapping->AddKeyFrame( frame );
    _local_mapping->Run();
    
    // 更新局部地图中的关键帧和地图点（用于追踪）
    _local_mapping->UpdateLocalKeyframes( frame );
    _local_mapping->UpdateLocalMapPoints( frame );
    
    _last_key_frame = frame; 
    _processed_frames =0;
    
}

// TODO 考虑sparse alignment 失败的情况
bool VisualOdometry::TrackRefFrame()
{
    LOG(INFO) << "track ref frame"<<endl;
    _TCR_estimated = SE3();
    SE3 TCR = _TCR_estimated; 
    for ( int level = _curr_frame->_pyramid.size()-1; level>=0; level -- ) {
        
        _align[level].SetTCR(TCR);
        _align[level].SparseImageAlignmentCeres( _ref_frame, _curr_frame, level );
        TCR = _align[level].GetEstimatedT21();
        
        Mat ref_img = _ref_frame->_color.clone();
        Mat curr_img = _curr_frame->_color.clone();
        
        /*
        for ( auto& obs_pair: _ref_frame->_obs ) {
            Vector3d pt_ref = _ref_frame->_camera->Pixel2Camera( obs_pair.second.head<2>(), obs_pair.second[2] );
            Vector2d px_curr = _curr_frame->_camera->Camera2Pixel( TCR * pt_ref );
            cv::circle( ref_img, cv::Point2f( obs_pair.second[0], obs_pair.second[1]), 2, cv::Scalar(0,250,0), 2);
            cv::circle( curr_img, cv::Point2f(px_curr[0], px_curr[1]), 2, cv::Scalar(0,250,0), 2);
        }
        cv::imshow("alignment ref", ref_img);
        cv::imshow("alignment curr", curr_img);
        cv::waitKey(1);
        */
    }
    
    if ( TCR.log().norm() > _options.max_sparse_align_motion ) {
        LOG(WARNING)<<"relative motion is too large, discard this frame. "<<endl;
        return false;
    }
    
    _TCR_estimated = TCR;
    
    // set the pose of current frame 
    _curr_frame->_T_c_w = _TCR_estimated * _ref_frame->_T_c_w;
    
    /*
    // 根据 sparse align 的结果，把参考帧中的地图点投影到当毅前上
    for ( auto& obs_pair: _ref_frame->_obs ) {
        MapPoint* mp = Memory::GetMapPoint( obs_pair.first );
        if ( mp->_bad ) 
            continue; 
        Vector3d pt_ref = _ref_frame->_camera->Pixel2Camera(obs_pair.second.head<2>(), obs_pair.second[2] );
        Vector3d pt_curr = _TCR_estimated * pt_ref;
        Vector2d px_curr = _curr_frame->_camera->Camera2Pixel( pt_curr );
        _curr_frame->_obs[ obs_pair.first ] = Vector3d( px_curr[0], px_curr[1], pt_curr[2]);
    }
    */
    
    return true; 
}

bool VisualOdometry::TrackLocalMap() 
{
    // Track Local Map 还是有点微妙的
    // 当前帧不是关键帧时，它不会提取新的特征，所以特征点都是从局部地图中投影过来的
    
    bool success = _local_mapping->TrackLocalMap( _curr_frame );
    
    /*
    // project the map points in last key frame into current 
    cv::Mat img_show = _curr_frame->_color.clone();
    for ( unsigned long& map_point_id: _last_key_frame->_map_point ) {
        MapPoint::Ptr mp = Memory::GetMapPoint( map_point_id );
        if ( mp->_bad ) continue; 
        Vector2d px_curr = _curr_frame->_camera->World2Pixel( mp->_pos_world, _curr_frame->_T_c_w );
        cv::circle( img_show, cv::Point2f(px_curr[0], px_curr[1]), 5, cv::Scalar(250,0,250), 2);
    }
    cv::imshow("current vs key-frame", img_show);
    cv::waitKey(1);
    */
    
    return success;
}

void VisualOdometry::ReprojectMapPointsInInitializaion()
{
    // 遍历地图中的3D点
    SE3 T12 = _ref_frame->_T_c_w*_curr_frame->_T_c_w.inverse();
    for ( auto iter = Memory::_points.begin(); iter!=Memory::_points.end();) {
        MapPoint* map_point = iter->second;
        bool bad_reproj = false, bad_depth = false;
        for ( auto& observation: map_point->_obs ) {
            Frame* frame = Memory::GetFrame(observation.first);
            Vector3d pt = frame->_camera->World2Camera( map_point->_pos_world, frame->_T_c_w );
            Vector2d px = frame->_camera->Camera2Pixel( pt );
            double reproj_error = (px - observation.second.head<2>()).norm();
            
            // LOG(INFO) << "pt[2] = " << pt[2]<<endl;
            
            if ( reproj_error > _options.init_reproj_error_th ) // 重投影误差超过阈值,说明是误匹配
            {
                LOG(INFO) << "bad reprojection error = "<<reproj_error<<endl;
                bad_reproj = true;
                
                /*
                Mat ref_img = _ref_frame->_color.clone();
                Mat curr_img = _curr_frame->_color.clone();
                cv::circle( ref_img, cv::Point2f( map_point->_obs[0][0], map_point->_obs[0][1]), 2, cv::Scalar(0,0,250), 2);
                cv::circle( curr_img, cv::Point2f( map_point->_obs[1][0], map_point->_obs[1][1]), 2, cv::Scalar(0,0,250), 2);
                cv::imshow("wrong point ref", ref_img);
                cv::imshow("wrong point curr", curr_img);
                cv::waitKey(0);
                */
                break;
            }
            
            if ( pt[2] < 0.001 || pt[2] > 1000 ) {
                // 距离太近或太远，可能原因是两条射线平行
                LOG(INFO) << "bad depth = "<<pt[2]<<" in frame "<<observation.first<<", reproj = "<<reproj_error<<endl;
                bad_depth = true;
                
                /*
                // plot the bad one 
                Mat ref_img = _ref_frame->_color.clone();
                Mat curr_img = _curr_frame->_color.clone();
                cv::circle( ref_img, cv::Point2f( map_point->_obs[0][0], map_point->_obs[0][1]), 2, cv::Scalar(0,0,250), 2);
                cv::circle( curr_img, cv::Point2f( map_point->_obs[1][0], map_point->_obs[1][1]), 2, cv::Scalar(0,0,250), 2);
                cv::imshow("wrong point ref", ref_img);
                cv::imshow("wrong point curr", curr_img);
                cv::waitKey(0);
                */
                
                break;
            }
        }
        
        if ( bad_reproj == true ) {
            LOG(INFO) << "set point "<<iter->first<<" as bad one"<<endl;
            iter->second->_bad = true;
            iter++;
            // iter = Memory::_points.erase( iter ); // 别随便删啊。。。删的话要把observation都去掉
        } else if ( bad_depth ) {
            iter->second->_bad = true;
            LOG(INFO) << "set point "<<iter->first<<" as bad one"<<endl;
            iter++;
        } else {
            iter++;
        }
    }
}

bool VisualOdometry::NeedNewKeyFrame()
{
    // 天哪orb的keyframe选择策略是怎么想出来的，莫非用机器学习去学的？
    // 话说这事还真能用机器学习学吧...
    // 先简单点，根据上个key-frame和当前的位姿差，以及tracked points做决定
    
    // 条件1，关键帧不能太密集，中间要经过一定的帧数
    if ( _processed_frames < 5 ) 
        return false; 
    
    SE3 deltaT = _last_key_frame->_T_c_w.inverse()*_curr_frame->_T_c_w;
    
    // 度量旋转李代数和平移向量的范数
    double dRotNorm = deltaT.so3().log().norm();
    double dTransNorm = deltaT.translation().norm();
    
    LOG(INFO) << "rot = "<<dRotNorm << ", t = "<< dTransNorm<<endl;
    if ( dRotNorm < _min_keyframe_rot && dTransNorm < _min_keyframe_trans )     // 平移或旋转都很小
        return false; 
    
    // 看跟踪是否快挂了
    if ( _curr_frame->_obs.size() < 30 )
        return true;
    
    // 计算正确跟踪的特征数量
    // 如果这个数量太少，很可能前面计算结果也是不对的
    return true; 
}

void VisualOdometry::ResetCurrentObservation()
{
    // set the observation in current and ref 
    double mean_depth = 0; 
    int cnt = 0;
    auto& all_points = Memory::GetAllPoints();
    // 注意这里还没有归一化，所以深度可能会有很大的值
    for ( auto iter = all_points.begin(); iter!=all_points.end(); ) {
        
        MapPoint* map_point = iter->second;
        Vector3d pt_ref = _ref_frame->_camera->World2Camera( map_point->_pos_world, _ref_frame->_T_c_w );
        Vector3d pt_curr = _curr_frame->_camera->World2Camera( map_point->_pos_world, _curr_frame->_T_c_w );
        
        if ( pt_ref[2] <0 || pt_ref[2]>1000 || pt_curr[2] <0 || pt_curr[2]>1000 ) {
            LOG(INFO) << "bad depth, pt_ref[2] = " << pt_ref[2] <<" and pt_curr[2] = " << pt_curr[2] <<endl;
            iter->second->_bad = true;
            iter++;
            continue; 
        } else {
            iter->second->_bad = false;
            iter++;
            mean_depth += pt_ref[2];
            cnt++;
            mean_depth += pt_curr[2];
            cnt++;
        }
    }
    
    mean_depth /= cnt;
    LOG(INFO) << "mean depth = " << mean_depth << endl;
    double scale = 1.0 / mean_depth;
    
    Vector3d t = _curr_frame->_T_c_w.translation();
    _curr_frame->_T_c_w.translation() = t*scale;
    
    for ( auto& map_point_pair: Memory::GetAllPoints() ) {
        MapPoint* map_point = map_point_pair.second;
        map_point->_pos_world = map_point->_pos_world * scale;
    }
    
    // set the observation to reprojected point 
    for ( auto& map_point_pair: Memory::GetAllPoints() ) {
        MapPoint* map_point = map_point_pair.second;
        for ( auto& obs_pair:map_point->_obs ) {
            if ( obs_pair.first == _ref_frame->_id ) {
                Vector3d pt = _ref_frame->_camera->World2Camera( map_point->_pos_world, _ref_frame->_T_c_w );
                Vector2d px = _ref_frame->_camera->World2Pixel( map_point->_pos_world, _ref_frame->_T_c_w );
                _ref_frame->_obs[ map_point_pair.first ] = Vector3d( px[0], px[1], pt[2] );
                // obs_pair.second = Vector3d(px[0], px[1], pt[2]) ;
            } else {
                Vector3d pt = _curr_frame->_camera->World2Camera( map_point->_pos_world, _curr_frame->_T_c_w );
                Vector2d px = _curr_frame->_camera->World2Pixel( map_point->_pos_world, _curr_frame->_T_c_w );
                _curr_frame->_obs[ map_point_pair.first ] = Vector3d( px[0], px[1], pt[2] );
                // obs_pair.second = Vector3d(px[0], px[1], pt[2]) ;
            }
        }
    }
}

bool VisualOdometry::CheckInitializationByDescriptors()
{
    // 计算两个帧中特征点的描述
    _orb_extractor->Compute( _curr_frame );
    
    LOG(INFO) << _ref_frame->_map_point_candidates.size()<<","<<_curr_frame->_map_point_candidates.size()<<endl;
    
    vector<pair<int,int>> matches;
    int good_pts = _orb_matcher->CheckFrameDescriptors( _ref_frame, _curr_frame, matches ); 
    LOG(INFO) << "good features: "<<good_pts<<endl;
    
    // 对成功匹配的点进行三角化
    SE3 T21 = _curr_frame->_T_c_w * _ref_frame->_T_c_w.inverse();
    for ( auto& m:matches ) {
        int i1 = m.first;
        int i2 = m.second;
        
        assert( _ref_frame->_candidate_status[i1] == CandidateStatus::WAIT_TRIANGULATION );
        assert( _curr_frame->_candidate_status[i2] == CandidateStatus::WAIT_TRIANGULATION );
        
        Vector3d pt_ref = _ref_frame->_camera->Pixel2Camera(
            Vector2d(_ref_frame->_map_point_candidates[i1].pt.x, _ref_frame->_map_point_candidates[i1].pt.y) );
        Vector3d pt_curr = _curr_frame->_camera->Pixel2Camera(
            Vector2d(_curr_frame->_map_point_candidates[i2].pt.x, _curr_frame->_map_point_candidates[i2].pt.y) );
        double depth1, depth2; 
        
        /*
        Mat ref_show = _ref_frame->_color.clone();
        Mat curr_show = _curr_frame->_color.clone();
        
        cv::circle( ref_show, _ref_frame->_map_point_candidates[i1].pt, 2, cv::Scalar(0,250,0), 2 );
        cv::circle( curr_show, _curr_frame->_map_point_candidates[i2].pt, 2, cv::Scalar(0,250,0), 2 );
        cv::imshow("ref", ref_show);
        cv::imshow("curr", curr_show);
        cv::waitKey(0);
        */
        
        if ( utils::DepthFromTriangulation( T21, pt_ref, pt_curr, depth1, depth2, 1e-4 ) ) {
            LOG(INFO) << "Create new map point" ;
            MapPoint* mp = Memory::CreateMapPoint();
            mp->_pos_world = _ref_frame->_camera->Camera2World( pt_ref*depth1, _ref_frame->_T_c_w );
            
            // set the observations 
            
            mp->_obs[_ref_frame->_id] = Vector3d( 
                _ref_frame->_map_point_candidates[i1].pt.x, 
                _ref_frame->_map_point_candidates[i1].pt.y, 
                pt_ref[2] 
            );
            _ref_frame->_obs[mp->_id] = mp->_obs[_ref_frame->_id];
            
            mp->_obs[_curr_frame->_id] = Vector3d( 
                _curr_frame->_map_point_candidates[i2].pt.x, 
                _curr_frame->_map_point_candidates[i2].pt.y, 
                pt_curr[2] 
            );
            _curr_frame->_obs[mp->_id] = mp->_obs[_curr_frame->_id];
            
            // mp->PrintInfo();
            
            // other statistics 
            mp->_cnt_found = 2;
            mp->_cnt_visible = 2;
            
            mp->_first_observed_frame = _ref_frame->_id;
            mp->_last_seen = _curr_frame->_id;
            mp->_track_in_view = true;
            
            mp->_descriptors.push_back(_ref_frame->_descriptors[i1]);
            mp->_descriptors.push_back(_curr_frame->_descriptors[i2]);
            
            _ref_frame->_candidate_status[i1] = CandidateStatus::TRIANGULATED;
            _curr_frame->_candidate_status[i2] = CandidateStatus::TRIANGULATED;
            
            mp->ComputeDistinctiveDesc();
            
            _ref_frame->_triangulated_mappoints[i1] = mp;
            _curr_frame->_triangulated_mappoints[i2] = mp;
            
        } else {
            // 三角化失败，可能因为平行性太强，但是不能把它设成BAD，可以等待之后再进行匹配
            _ref_frame->_candidate_status[i1] = CandidateStatus::WAIT_TRIANGULATION;
            _curr_frame->_candidate_status[i2] = CandidateStatus::WAIT_TRIANGULATION;
        }
    }
    
    // show the inliers 
    Mat show(_ref_frame->_color.rows, 2*_ref_frame->_color.cols, CV_8UC3);
    _ref_frame->_color.copyTo( show(cv::Rect(0,0,show.cols/2, show.rows )));
    _curr_frame->_color.copyTo( show(cv::Rect(show.cols/2,0, show.cols/2, show.rows)));
    
    for ( size_t i1=0; i1<_ref_frame->_map_point_candidates.size(); i1++ ) {
        if ( _ref_frame->_candidate_status[i1] == CandidateStatus::TRIANGULATED ) {
            cv::circle( show, _ref_frame->_map_point_candidates[i1].pt, 2, cv::Scalar(0,250,0),2 );
            //cv::line( show, _ref_frame->_map_point_candidates[i1].pt, _curr_frame->_map_point_candidates[i2].pt+cv::Point2f(show.cols/2, 0), cv::Scalar(0,250,0) );
        } else if ( _ref_frame->_candidate_status[i1] == CandidateStatus::WAIT_TRIANGULATION ) {
            cv::circle( show, _ref_frame->_map_point_candidates[i1].pt, 2, cv::Scalar(250,0,0),2 );
        } else {
            cv::circle( show, _ref_frame->_map_point_candidates[i1].pt, 2, cv::Scalar(0,0,250),2 );
        }
    }
    
    for ( size_t i2=0; i2<_curr_frame->_map_point_candidates.size(); i2++ ) {
        if ( _curr_frame->_candidate_status[i2] == CandidateStatus::TRIANGULATED ) {
            cv::circle( show, _curr_frame->_map_point_candidates[i2].pt+cv::Point2f(show.cols/2,0), 2, cv::Scalar(0,250,0),2 );
        } else if ( _curr_frame->_candidate_status[i2] == CandidateStatus::WAIT_TRIANGULATION ) {
            cv::circle( show, _curr_frame->_map_point_candidates[i2].pt+cv::Point2f(show.cols/2,0), 2, cv::Scalar(250,0,0),2 );
        } else {
            cv::circle( show, _curr_frame->_map_point_candidates[i2].pt+cv::Point2f(show.cols/2,0), 2, cv::Scalar(0,0,250),2 );
        }
    }
    
    cv::imshow("feature match", show);
    cv::waitKey(0);
}

bool VisualOdometry::CheckObservationByDescriptors()
{
    // 将 current 的 observation 设成它的特征点
    for ( auto& obs_pair: _curr_frame->_obs ) {
        _curr_frame->_map_point_candidates.push_back( 
            cv::KeyPoint( obs_pair.second[0], obs_pair.second[1], 7 )
        );
        _curr_frame->_candidate_status.push_back( CandidateStatus::WAIT_DESCRIPTOR );
    }
    
    _orb_extractor->Compute( _curr_frame );
    
    // compare these descriptors to map points 
    int i=0; 
    for ( auto iter = _curr_frame->_obs.begin(); iter!=_curr_frame->_obs.end(); i++ ) {
        MapPoint* mp = Memory::GetMapPoint( iter->first );
        int distance = ORBMatcher::DescriptorDistance( _curr_frame->_descriptors[i], mp->_distinctive_desc );
        
        LOG(INFO) << "distance = "<<distance<<endl;
        
        if ( distance < 80 ) {
            // take this as a good one  
            mp->_descriptors.push_back( _curr_frame->_descriptors[i] );
            mp->ComputeDistinctiveDesc();
            _curr_frame->_candidate_status[i] = CandidateStatus::TRIANGULATED;
            mp->_obs[_curr_frame->_id] = iter->second;
            mp->_descriptors.push_back( _curr_frame->_descriptors[i] );
            mp->ComputeDistinctiveDesc();
            iter++;
        } else {
            
            // observation does not match descriptors 
            iter = _curr_frame->_obs.erase(iter);
            _curr_frame->_candidate_status[i] = CandidateStatus::BAD;
        }
    }
    
    // _curr_frame->_map_point_candidates.clear();
    // _curr_frame->_descriptors.clear();
    
    LOG(INFO) << "good observations: "<< _curr_frame->_obs.size()<<endl;
    
    cv::Mat img_show = _curr_frame->_color.clone();
    for ( auto iter=_curr_frame->_obs.begin(); iter!=_curr_frame->_obs.end(); iter++ ) {
        // reset the observation 
        cv::circle( img_show, cv::Point2f( (iter->second)[0], (iter->second)[1] ), 2, cv::Scalar(0,0,250), 2 );
    }
    cv::imshow("good observations" , img_show);
    cv::waitKey(0);
    
}

    

}
