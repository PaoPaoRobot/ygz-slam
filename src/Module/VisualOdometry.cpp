#include "ygz/Basic.h"
#include "ygz/Algorithm.h"
#include "ygz/Module/VisualOdometry.h"
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
        SetKeyframe( frame );
        _status = VO_INITING;
        _tracker->SetReference( _ref_frame );
        return true;
    }
    
    _curr_frame = frame ;
    _last_status = _status; 
    
    if ( _status == VO_INITING )
    {
        bool success = MonocularInitialization();
        if ( success != VO_GOOD )       // 光流跟踪丢了
            return false; 
        else 
            _status = VO_GOOD;
    }
    else 
    {
        bool OK = false;
        if ( _status == VO_GOOD )
        {
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
                _ref_frame = _curr_frame;
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
                
                // initialization is good, set the map points
                CreateMapPointsAfterMonocularInitialization(
                    features_ref, pixels_curr, 
                    pts_ref_triangulated, inliers
                );
                
                // set current frame as a key-frame 
                SetKeyframe( _curr_frame );
                
                return true; 
            } // endif init_success 
        } // endif min_disparity>_options._min_init_disparity
    }
    
    return false;
}
    
void VisualOdometry::SetKeyframe(Frame* frame)
{
    // 向memory注册这个关键帧，提取新的特征点，更新Local Keyframes and local map points 
    frame->_is_keyframe = true;
    frame = Memory::RegisterKeyFrame( frame );
    
    _detector->Detect( frame );
    _last_key_frame = frame;
}

void VisualOdometry::CreateMapPointsAfterMonocularInitialization(
    vector< Feature* >& features_ref, 
    vector< Vector2d >& pixels_curr, 
    vector< Vector3d >& pts_triangulated, 
    vector< bool >& inliers )
{
}

bool VisualOdometry::TrackRefFrame()
{
    // 使用 matcher 的sparse alignment 追踪参考帧中的
}
    

    
}