#include "ygz/tracker.h"
#include <opencv2/video/video.hpp>

namespace ygz {
    
Tracker::Tracker( ) : 
_detector( new FeatureDetector )
{
    // read the params from config 
    _min_features_initializing = Config::get<int>("init.min_features");
    
}

void Tracker::SetReference(Frame::Ptr ref)
{
    // detect the features in reference 
    _detector->Detect( ref );
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
    for ( MapPoint p: _ref->_map_point_candidates ) {
        _px_ref.push_back( cv::Point2f(p._pos_tracked[0], p._pos_tracked[1]) );
    }
    
    _px_curr = _px_ref;
    LOG(INFO) << "Keypoints detected: "<<_px_ref.size()<<endl;
}

void Tracker::Track(Frame::Ptr curr)
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
        if ( !status[i] )
        {
            px_ref_it = _px_ref.erase( px_ref_it );
            continue;
        }
        _px_curr.push_back( *pt_cur_it );
        px_ref_it++;
    }
    
    LOG(INFO) << "pts tracked in LK: " << _px_curr.size() << endl;
}


Tracker::~Tracker()
{

}

void Tracker::PlotTrackedPoints() const
{
    Mat img_show;
    if ( _curr != nullptr) {
        img_show = _curr->_color.clone(); 
    } else {
        return; 
    }
    
    for ( const cv::Point2f& p: _px_curr ) {
        cv::circle( img_show, p, 5, cv::Scalar(0,250,0), 2 );
    }
    cv::imshow( "tracked points", img_show );
    cv::waitKey(0);
}

float Tracker::MeanDisparity() const
{
    auto ref_it = _px_ref.begin();
    auto curr_it = _px_curr.begin();
    
    double mean_disparity; 
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