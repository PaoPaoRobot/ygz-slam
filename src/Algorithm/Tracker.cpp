#include "ygz/Algorithm/Tracker.h"

#include <opencv2/video/video.hpp> // for KLT 

namespace ygz 
{
    
Tracker::Tracker()
{
    _option._min_feature_tracking = Config::Get<int>("tracker.min_features");
}

void Tracker::SetReference(Frame* ref)
{
    if ( ref->_features.size() < _option._min_feature_tracking )
    {
        LOG(WARNING) << "Track a reference with little features: "<<ref->_features.size()<<", abort."<<endl;
        _status = TRACK_NOT_READY;
        return;
    }
    
    // set the reference 
    _ref = ref;
    _curr = ref;
    _status = TRACK_GOOD;
    
    for ( Feature* fea: ref->_features )
    {
        _tracked_features.push_back( fea );
        _px_curr.push_back( cv::Point2f(fea->_pixel[0], fea->_pixel[1]) );
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
    LOG(INFO) << "Tracked pixels: " << _px_curr.size();
    
    if ( _px_curr.size() < _option._min_feature_tracking )
    {
        _status = TRACK_LOST;
        LOG(WARNING) << "Track with little features, set it as lost."<<endl;
    }
}

void Tracker::GetTrackedPixel(
    vector< Feature* >& feature1, 
    vector< Vector2d >& pixels2) const
{
    for ( Feature* fea: _tracked_features )
        feature1.push_back( fea );
    for ( auto px: _px_curr )
        pixels2.push_back( Vector2d(px.x,px.y) );
}

void Tracker::TrackKLT()
{
    vector<uchar> status;
    vector<float> error;
    vector<float> min_eig_vec;
    cv::TermCriteria termcrit ( 
        cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 
        _option.klt_max_iter, 
        _option.klt_eps 
    );
    
    // convert the list to vector 
    vector<size_t> tracked_index;
    vector<cv::Point2f> pt_ref; 
    
    for ( Feature* fea: _tracked_features )
    {
        pt_ref.push_back( cv::Point2f(fea->_pixel[0], fea->_pixel[1]) );
    }
    
    vector<cv::Point2f> pt_curr; 
    for ( cv::Point2f& p: _px_curr )  {
        pt_curr.push_back( p );
    }
    
    LOG(INFO) << "pt cur = "<< pt_curr.size()<<endl;
    
    cv::calcOpticalFlowPyrLK ( 
        _ref->_pyramid[0], _curr->_pyramid[0],
        pt_ref, pt_curr,
        status, error,
        cv::Size2i ( _option.klt_win_size, _option.klt_win_size ),
        4, termcrit, cv::OPTFLOW_USE_INITIAL_FLOW 
    );
    
    // copy the successfully tracked features
    _px_curr.clear();
    
    size_t iStatus = 0;
    for ( auto iter=_tracked_features.begin(); iter!=_tracked_features.end();  iStatus++ )
    {
        if ( !status[iStatus] || ! _curr->InFrame(pt_curr[iStatus], 20) ) {
            iter = _tracked_features.erase(iter);
        } else {
            iter++;
            _px_curr.push_back( pt_curr[iStatus] );
        }
    }
}

float Tracker::MeanDisparity() const
{
    assert( _tracked_features.size() == _px_curr.size() );
    
    float mean_disparity = 0;
    auto iter_ref = _tracked_features.begin();
    size_t iCur = 0;
    for ( ; iter_ref!=_tracked_features.end(); iter_ref++, iCur++ )
    {
        mean_disparity += ((*iter_ref)->_pixel - Vector2d(_px_curr[iCur].x, _px_curr[iCur].y)).norm();
    }
    return mean_disparity/_tracked_features.size();
}

void Tracker::PlotTrackedPoints() const 
{
    if ( _curr == nullptr )
        return;
    Mat ref_show = _ref->_color.clone();
    Mat curr_show = _curr->_color.clone();
    
    auto curr_it = _px_curr.begin();
    
    int iCur = 0;
    for ( Feature* fea: _tracked_features )
    {
        cv::circle( ref_show, cv::Point2f(fea->_pixel[0], fea->_pixel[1]), 2, cv::Scalar(250,0,0), 2 );
        cv::circle( curr_show, _px_curr[iCur], 2, cv::Scalar(0,250,0), 2 );
        iCur++;
    }
    
    cv::imshow( "tracked points in ref", ref_show );
    cv::imshow( "tracked points in curr", curr_show );
    cv::waitKey(0);
}
    
    
    
}