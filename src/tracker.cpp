#include <opencv2/video/video.hpp>

#include "ygz/config.h"
#include "ygz/tracker.h"
#include "ygz/feature_detector.h"
#include "ygz/map_point.h"
#include "ygz/camera.h"

namespace ygz {
    
Tracker::Tracker( )
{
    // read the params from config 
    _min_features_initializing = Config::get<int>("init.min_features");
}

void Tracker::SetReference(Frame* ref)
{
    // detect the features in reference 
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
        _px_curr.push_back( p.pt );
    }
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
    
    // convert the list to vector 
    vector<size_t> tracked_index;
    vector<cv::Point2f> pt_ref; 
    for ( size_t i=0; i< _ref->_map_point_candidates.size(); i++ ) {
        if ( _ref->_candidate_status[i] == CandidateStatus::BAD ) 
            continue;
        pt_ref.push_back( _ref->_map_point_candidates[i].pt );
        tracked_index.push_back(i);
    }
    LOG(INFO) << "pt ref = "<<pt_ref.size()<<endl;
    
    vector<cv::Point2f> pt_curr; 
    for ( cv::Point2f& p: _px_curr )  {
        pt_curr.push_back( p );
    }
    
    LOG(INFO) << "pt cur = "<< pt_curr.size()<<endl;
    
    cv::calcOpticalFlowPyrLK ( _ref->_pyramid[0], _curr->_pyramid[0],
                               pt_ref, pt_curr,
                               status, error,
                               cv::Size2i ( klt_win_size, klt_win_size ),
                               4, termcrit, cv::OPTFLOW_USE_INITIAL_FLOW );
    
    _px_curr.clear();
    // copy the tracked ones 
    
    for ( size_t iStatus=0; iStatus<status.size(); ++iStatus )
    {
        if ( !status[iStatus] || ! _curr->InFrame(pt_curr[iStatus], 20) ) {
            _ref->_candidate_status[ tracked_index[iStatus] ] = CandidateStatus::BAD;
        } else {
            _px_curr.push_back( pt_curr[iStatus] );
        }
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
    auto curr_it = _px_curr.begin();
    
    for ( size_t i=0; i<_ref->_map_point_candidates.size(); i++ ) {
        if ( _ref->_candidate_status[i] == CandidateStatus::BAD ) 
            continue; 
        
        cv::circle( ref_show, _ref->_map_point_candidates[i].pt, 2, cv::Scalar(250,0,0), 2 );
        cv::circle( img_show, *curr_it, 2, cv::Scalar(0,250,0), 2 );
        curr_it++;
    }
    cv::imshow( "tracked points in curr", img_show );
    cv::imshow( "tracked points in ref", ref_show );
    cv::waitKey(0);
}

float Tracker::MeanDisparity() const
{
    auto ref_it = _ref->_map_point_candidates.begin();
    auto curr_it = _px_curr.begin();
    
    double mean_disparity =0; 
    for ( size_t i=0; i<_ref->_map_point_candidates.size(); i++ ) {
        if ( _ref->_candidate_status[i] == CandidateStatus::BAD ) 
            continue;
        mean_disparity += Vector2d( _ref->_map_point_candidates[i].pt.x-curr_it->x, _ref->_map_point_candidates[i].pt.y - curr_it->y).norm();
        curr_it++;
    }
    return (float) mean_disparity/_ref->_map_point_candidates.size();
}

void Tracker::GetTrackedPointsNormalPlane(
    vector< Vector2d >& pt1, 
    vector< Vector2d >& pt2) const
{
    for ( auto& px:_ref->_map_point_candidates ) {
        Vector2d v(px.pt.x, px.pt.y);
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
    for ( size_t i1=0; i1<_ref->_map_point_candidates.size(); i1++ ) {
        if ( _ref->_candidate_status[i1] == CandidateStatus::BAD) 
            continue;
        px1.push_back( Vector2d( _ref->_map_point_candidates[i1].pt.x, _ref->_map_point_candidates[i1].pt.y) );
    }
    for ( auto& px:_px_curr ) {
        px2.push_back( Vector2d(px.x, px.y) );
    }
}

    
}