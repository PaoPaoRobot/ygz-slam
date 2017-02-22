#include "ygz/Basic.h"
#include "ygz/Algorithm.h"
#include "ygz/Module/VisualOdometry.h"
#include "ygz/Module/LocalMapping.h"
#include "ygz/System/System.h"

namespace ygz 
{
    
VisualOdometry::VisualOdometry( System* system )
: _system( system )
{
    _options._min_keyframe_rot = Config::Get<double>("vo.keyframe.min_rot");
    _options._min_keyframe_trans = Config::Get<double>("vo.keyframe.min_trans");
    _options._min_keyframe_features = Config::Get<int>("vo.keyframe.min_features");
    
    // TODO 为了调试方便所以在这里写new，但最后需要挪到system里面管理
    _init = new Initializer();
    _detector = new FeatureDetector();
    _detector->LoadParams();
    _tracker = new Tracker();
    _matcher = new Matcher();
    _local_mapping = new LocalMapping();
}
    
VisualOdometry::~VisualOdometry()
{
    if ( _init )
        delete _init;
    if ( _detector )
        delete _detector;
    if ( _tracker )
        delete _tracker;
    if ( _matcher )
        delete _matcher;
}

bool VisualOdometry::AddFrame( Frame* frame )
{
    if ( _status == VO_NOT_READY )
    {
        // case 1: set first frame 
        _ref_frame = frame;
        SetKeyframe( frame );   // this will detect features on frame
        _status = VO_INITING;
        _tracker->SetReference( _ref_frame );
        return true;
    }
    
    _curr_frame = frame ;
    _last_status = _status; 
    
    if ( _status == VO_INITING )
    {
        bool success = MonocularInitialization();
        if ( success == false )       // 光流跟踪丢了
            return false; 
        else 
            _status = VO_GOOD;
    }
    else 
    {
        bool OK = false;
        if ( _status == VO_GOOD )
        {
            _curr_frame->_TCW = _ref_frame->_TCW;   // set the last pose as initial value
            OK = TrackRefFrame(); // sparse image alignment 
            if ( OK )
            {
                // alignment from local map points
                OK = TrackLocalMap(); 
            }
            
            if ( OK )
            {
                _status = VO_GOOD;
                
                // track good, check new key-frame 
                if ( NeedNewKeyFrame() )
                {
                    // 当前帧是关键帧
                    SetKeyframe( _curr_frame );
                }
                else 
                {
                    // 普通帧
                }
                if ( _ref_frame->_is_keyframe == false )
                    delete _ref_frame;
                _ref_frame = _curr_frame;
                
                _processed_frames++;
            }
            else 
            {
                // abort this frame 
                _status = VO_LOST;
                return false;
            }
        }// endif _status == VO_GOOD 
        else 
        {
            // try relocalization 
        }
    }
    return true;
}

// 初始化部分
// 用tracker去追踪参考帧里的特征点，然后用initializer初始化
bool VisualOdometry::MonocularInitialization() 
{
    if ( _tracker->Status() == Tracker::TRACK_NOT_READY )
    {
        _tracker->SetReference( _curr_frame );
    }
    else if ( _tracker->Status() == Tracker::TRACK_GOOD )
    {
        _tracker->Track( _curr_frame );
        float min_disparity = _tracker->MeanDisparity();
        if ( min_disparity > _options._min_init_disparity )
        {
            vector<Feature*> features_ref;
            vector<Vector2d> pixels_curr;
            _tracker->GetTrackedPixel( features_ref, pixels_curr);
            assert( features_ref.size() == pixels_curr.size() );
            
            vector<Vector2d> pixels_ref; 
            for ( Feature* fea: features_ref )
                pixels_ref.push_back( fea->_pixel );
            
            // try initialize 
            bool init_success = _init->TryInitialize(
                pixels_ref, pixels_curr,
                _ref_frame, _curr_frame
            );
            
            if ( init_success )
            {
                // 初始化算法认为初始化通过，但可能会有outlier和误跟踪
                // 光流在跟踪之后不能保证仍是角点
                SE3 T21 = _init->GetT21();
                vector<Vector3d> pts_ref_triangulated;
                vector<bool> inliers; 
                _init->GetTriangluatedPoints( pts_ref_triangulated, inliers );
                
                // 做一遍BA以减少噪声的影响
                ba::TwoViewBACeres( 
                    _ref_frame->_TCW, T21, 
                    pixels_ref, pixels_curr, 
                    inliers, pts_ref_triangulated );
                
                // count the inliers 
                int cnt_inliers =0; 
                for ( bool in:inliers )
                    cnt_inliers++;
                if ( cnt_inliers < _options._min_init_features )
                {
                    return false;
                }
                
                _curr_frame->_TCW = T21;
                
                // initialization is good, set the map points
                CreateMapPointsAfterMonocularInitialization(
                    features_ref, pixels_curr, 
                    pts_ref_triangulated, inliers
                );
                
                // set current frame as a key-frame 
                SetKeyframe( _curr_frame );
                
                _ref_frame = _curr_frame; 
                return true; 
            } // endif init_success 
        } // endif min_disparity>_options._min_init_disparity
    }
    
    return false;
}
    
void VisualOdometry::SetKeyframe(Frame* frame)
{
    // 向memory注册这个关键帧，提取新的特征点，更新Local Keyframes and local map points 
    assert(frame->_camera!=nullptr);
    frame->_is_keyframe = true;
    frame = Memory::RegisterKeyFrame( frame );
    LOG(INFO)<<"setting new keyframe "<<frame->_keyframe_id<<endl;
    // set the map point observation 
    for ( Feature* fea: frame->_features )
    {
        if ( !fea->_bad && fea->_mappoint && !fea->_mappoint->_bad )
        {
            MapPoint* mp = fea->_mappoint;
            mp->_last_seen = frame->_keyframe_id;
            mp->_obs[frame->_keyframe_id] = fea;
        }
    }
    _detector->ComputeAngleAndDescriptor( frame );
    // 新增特征点(2D)
    _detector->Detect( frame, false );
    /*
    assert( frame->_bow_vec.empty() );
    frame->ComputeBoW();
    
    LOG(INFO)<<"this key-frame has "<<frame->_features.size()<<" features."<<endl;
    
    _local_mapping->UpdateLocalKeyframes( frame );
    _local_mapping->UpdateLocalMapPoints( frame );
    
    _local_mapping->AddKeyFrame( frame );
    _local_mapping->Run();
    */
    _last_key_frame = frame;
    _processed_frames = 0;
}

void VisualOdometry::CreateMapPointsAfterMonocularInitialization(
    vector< Feature* >& features_ref, 
    vector< Vector2d >& pixels_curr, 
    vector< Vector3d >& pts_triangulated, 
    vector< bool >& inliers )
{
    // LOG(INFO)<<features_ref.size()<<","<<pixels_curr.size()<<","<<pts_triangulated.size()<<","<<inliers.size()<<endl;
    
    // compute the scale and create the map points 
    double mean_depth = 0;
    int cnt_valid = 0;
    
    for ( size_t i=0; i<features_ref.size(); i++ ) 
    {
        if ( inliers[i] == true ) 
        {
            // create the map points 
            MapPoint* mp = Memory::CreateMapPoint();
            mp->_cnt_found = mp->_cnt_visible = 2; 
            mp->_pos_world = pts_triangulated[i];
            mp->_last_seen = _curr_frame->_id;
            mp->_obs[_ref_frame->_id] = features_ref[i];
            features_ref[i]->_mappoint = mp;
            features_ref[i]->_depth = _ref_frame->_camera->World2Camera( mp->_pos_world, _ref_frame->_TCW )[2];
            
            // create the features in current 
            Feature* new_feature = new Feature(
                pixels_curr[i], features_ref[i]->_level, 
                features_ref[i]->_score
            );
            new_feature->_frame = _curr_frame;
            new_feature->_depth = _curr_frame->_camera->World2Camera( mp->_pos_world, _curr_frame->_TCW )[2];
            new_feature->_mappoint = mp;
            mp->_obs[_curr_frame->_id] = new_feature;
            _curr_frame->_features.push_back( new_feature );
            
            mean_depth += features_ref[i]->_depth;
            cnt_valid++;
        }
    }
    
    // rescale the map to keep the mean_depth=1 
    mean_depth /= cnt_valid;
    for ( Feature* fea : features_ref)
        if ( fea->_mappoint )
        {
            fea->_mappoint->_pos_world = fea->_mappoint->_pos_world/mean_depth;
            fea->_depth /= mean_depth;
        }
    for ( Feature* fea: _curr_frame->_features )
        if ( fea->_mappoint )
            fea->_depth /= mean_depth; 
        
    // LOG(INFO)<<"inlier features: "<<cnt_valid<<endl;
    // ... and the pose 
    _curr_frame->_TCW.translation() = _curr_frame->_TCW.translation()/mean_depth;
    
    // compute the angles and descriptors in current frame 
    _detector->ComputeAngleAndDescriptor( _curr_frame );
}

bool VisualOdometry::TrackRefFrame()
{
    // 使用 matcher 的 sparse alignment 追踪参考帧中的观测点
    bool ret = _matcher->SparseImageAlignment( _ref_frame, _curr_frame );
    if ( ret == false )
    {
        LOG(WARNING) << "Track Ref frame failed, using motion model."<<endl;
        // try using the TCR esitmated in last loop 
        _curr_frame->_TCW = _TCR_estimated * _ref_frame->_TCW;
        return false;
    }
    
    _TCR_estimated = _matcher->GetTCR();
    
    _curr_frame->_TCW = _TCR_estimated * _ref_frame->_TCW;
    LOG(INFO) << "current pose estimated by sparse alignment: \n"<<_curr_frame->_TCW.matrix()<<endl;
    
    // let's see the projections
    // PlotTrackRefFrameResults();
    
    return true;
}
    
bool VisualOdometry::NeedNewKeyFrame()
{
    if ( _processed_frames < 5 )
        return false;
    
    SE3 deltaT = _last_key_frame->_TCW.inverse()*_curr_frame->_TCW;
    double dRotNorm = deltaT.so3().log().norm();
    double dTransNorm = deltaT.translation().norm();
    
    LOG(INFO) << "rot = "<<dRotNorm << ", t = "<< dTransNorm<<endl;
    if ( dRotNorm < _options._min_keyframe_rot && dTransNorm < _options._min_keyframe_trans ) // 平移或旋转都很小
        return false; 
    
    if ( _curr_frame->_features.size() < 30 )   // 跟踪快挂了
        return true;
    
    return true;
}
    
bool VisualOdometry::TrackLocalMap() 
{
    return _local_mapping->TrackLocalMap( _curr_frame );
}

bool VisualOdometry::CheckInitializationByDescriptors(
    vector< bool >& inliers)
{
    
    return true;
}

void VisualOdometry::PlotTrackRefFrameResults()
{
    Mat img_show = _curr_frame->_color.clone();
    Mat img_ref = _ref_frame->_color.clone();
    SE3 TCR = _matcher->GetTCR();
    for ( Feature* fea : _ref_frame->_features )
    {
        if ( fea->_mappoint && fea->_bad==false )
        {
            // Vector2d px = _curr_frame->_camera->World2Pixel( fea->_mappoint->_pos_world, _curr_frame->_TCW );
            Vector2d px = _curr_frame->_camera->World2Pixel( 
                _ref_frame->_camera->Pixel2Camera( fea->_pixel, fea->_depth ), TCR );
            cv::circle( img_show, cv::Point2f(fea->_pixel[0],fea->_pixel[1]), 1, cv::Scalar(0,0,250), 2);
            cv::circle( img_show, cv::Point2f(px[0],px[1]), 1, cv::Scalar(0,250,0), 2);
            cv::circle( img_ref, cv::Point2f(fea->_pixel[0],fea->_pixel[1]), 1, cv::Scalar(0,0,250), 2);
        }
    }
    
    cv::imshow("Track ref frame: curr", img_show );
    cv::imshow("Track ref frame: ref", img_ref );
    cv::waitKey(1);
}

    
}