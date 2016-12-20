#include <opencv2/video/video.hpp>

#include "ygz/config.h"
#include "ygz/tracker.h"
#include "ygz/feature_detector.h"
#include "ygz/map_point.h"

namespace ygz {
    
Tracker::Tracker( FeatureDetector* detector ) : 
_detector( detector )
{
    // read the params from config 
    _min_features_initializing = Config::get<int>("init.min_features");
}

void Tracker::SetReference(Frame::Ptr ref)
{
    // detect the features in reference 
    _detector->Detect( ref.get() );
    LOG(INFO) << "Keypoints detected: "<< ref->_map_point_candidates.size() <<endl;
    
    if ( ref->_map_point_candidates.size() < _min_features_initializing ) {
        LOG(WARNING) << "Init frame has few features, try moving in more textured environment. " << endl;
        _status = TRACK_NOT_READY;
        return; 
    }
    
    // good, set this frame to track 
    _ref = ref; 
    _curr = _ref;
    _status = TRACK_GOOD;
    
    // set the tracked pts in ref 
    for ( cv::KeyPoint& p: _ref->_map_point_candidates ) {
        _px_ref.push_back( p.pt );
    }
    _px_curr = _px_ref;
}

void Tracker::Track(Frame* curr)
{
    if ( _status == TRACK_NOT_READY ) {
        LOG(WARNING) << "reference is not ready, please set reference first! " << endl;
        return; 
    } else if ( _status == TRACK_LOST ) {
        LOG(WARNING) << "track is lost, please reset it" << endl;
        return;
    }
    
    _curr = curr; 
    TrackKLT();
    LOG(INFO) << "Pixels in current: " << _px_curr.size();
    
    if ( _px_curr.size() < 5 ) {
        // considered as track lost 
        _status = TRACK_LOST;
    }
}

void Tracker::TrackKLT()
{
    const double klt_win_size = 30.0;
    const int klt_max_iter = 30;
    const double klt_eps = 0.001;
    vector<uchar> status;
    vector<float> error;
    vector<float> min_eig_vec;
    cv::TermCriteria termcrit ( cv::TermCriteria::COUNT+cv::TermCriteria::EPS, klt_max_iter, klt_eps );
    
    LOG(INFO) << "pts to be tracked: "<< _px_ref.size() <<endl;
    // convert the list to vector 
    vector<cv::Point2f> pt_ref; 
    for ( cv::Point2f& p: _px_ref )  {
        pt_ref.push_back( p );
    }
    
    vector<cv::Point2f> pt_curr; 
    for ( cv::Point2f& p: _px_curr )  {
        pt_curr.push_back( p );
    }
    
    cv::calcOpticalFlowPyrLK ( _ref->_pyramid[0], _curr->_pyramid[0],
                               pt_ref, pt_curr,
                               status, error,
                               cv::Size2i ( klt_win_size, klt_win_size ),
                               4, termcrit, cv::OPTFLOW_USE_INITIAL_FLOW );
    
    LOG(INFO) << "KLT: ref pts = "<<_px_ref.size()<<", curr pts = "<<pt_curr.size()<<endl;
    
    auto px_ref_it = _px_ref.begin();
    auto pt_cur_it = pt_curr.begin();
    
    _px_curr.clear();
    // copy the tracked ones and remove the lost ones
    for ( size_t i=0; px_ref_it != _px_ref.end(); ++i, ++pt_cur_it )
    {
        if ( !status[i] || ! _curr->InFrame( *pt_cur_it) )
        {
            px_ref_it = _px_ref.erase( px_ref_it );
            continue;
        }
        // LOG(INFO) << "error = "<<error[i]<<endl;
        _px_curr.push_back( *pt_cur_it );
        px_ref_it++;
    }
    
    LOG(INFO) << "pts tracked in LK: " << _px_curr.size() << endl;
}


void Tracker::PlotTrackedPoints() const
{
    Mat img_show;
    if ( _curr != nullptr) {
        img_show = _curr->_color.clone(); 
    } else {
        return; 
    }
    
    Mat ref_show = _ref->_color.clone();
    
    auto ref_it = _px_ref.begin();
    auto curr_it = _px_curr.begin();
    
    LOG(INFO) << "ref pts: "<<_px_ref.size()<<", curr pts: "<<_px_curr.size();
    for ( ; curr_it != _px_curr.end(); ++ref_it, ++curr_it ) {
        cv::circle( img_show, *curr_it, 2, cv::Scalar(0,250,0), 2 );
        cv::circle( img_show, *ref_it, 2, cv::Scalar(250,0,0), 2 );
        cv::line( img_show, *ref_it, *curr_it, cv::Scalar(0,250,0), 2);
        
        cv::circle( ref_show, *curr_it, 2, cv::Scalar(0,250,0), 2 );
        cv::circle( ref_show, *ref_it, 2, cv::Scalar(250,0,0), 2 );
        cv::line( ref_show, *ref_it, *curr_it, cv::Scalar(0,250,0), 2);
    }
    
    cv::imshow( "tracked points in curr", img_show );
    // cv::imshow( "tracked points in ref", ref_show );
    cv::waitKey(1);
}

float Tracker::MeanDisparity() const
{
    auto ref_it = _px_ref.begin();
    auto curr_it = _px_curr.begin();
    
    double mean_disparity =0; 
    for ( auto ref_it_end = _px_ref.end(); ref_it!=ref_it_end; ref_it++, curr_it++ ) {
        mean_disparity += Vector2d(ref_it->x-curr_it->x, ref_it->y - curr_it->y).norm();
    }
    return (float) mean_disparity/_px_ref.size();
}

void Tracker::GetTrackedPointsNormalPlane(
    vector< Vector2d >& pt1, 
    vector< Vector2d >& pt2) const
{
    for ( auto& px:_px_ref ) {
        Vector2d v(px.x, px.y);
        pt1.push_back(_ref->_camera->Pixel2Camera( v ).head<2>() );
    }
    for ( auto& px:_px_curr ) {
        Vector2d v(px.x, px.y);
        pt2.push_back( _curr->_camera->Pixel2Camera( v ).head<2>() );
    }
}

void Tracker::GetTrackedPixel(
    vector< Vector2d >& px1, 
    vector< Vector2d >& px2) const
{
    for ( auto& px:_px_ref ) {
        Vector2d v(px.x, px.y);
        px1.push_back( Vector2d(px.x, px.y) );
    }
    for ( auto& px:_px_curr ) {
        Vector2d v(px.x, px.y);
        px2.push_back( Vector2d(px.x, px.y) );
    }
}

    
}