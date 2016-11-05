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
    // good, track this frame 
    _ref = ref; 
    _px_curr = _px_ref;
    _status = TRACK_GOOD;
    
    // set the tracked pts in ref 
    for ( MapPoint p: _ref->_map_point_candidates ) {
        _px_ref.push_back( cv::Point2f(p._pos_tracked[0], p._pos_tracked[1]) );
    }
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
    
    TrackKLT();
    
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
    
    vector<cv::Point2f> pt_curr; 
    
    cv::calcOpticalFlowPyrLK ( _ref->_pyramid[0], _curr->_pyramid[0],
                               _px_ref, pt_curr,
                               status, error,
                               cv::Size2i ( klt_win_size, klt_win_size ),
                               4, termcrit, cv::OPTFLOW_USE_INITIAL_FLOW );
    
    LOG(INFO) << "KLT: ref pts = "<<_px_ref.size()<<", curr pts = "<<pt_curr.size()<<endl;
    auto pt_ref_it = _px_ref.begin();
    auto pt_cur_it = pt_curr.begin();
    
    // select the tracked ones 
    for ( size_t i=0; pt_ref_it != _px_ref.end(); ++i )
    {
        if ( !status[i] )
        {
            continue;
        }
        
        _px_curr.push_back( *pt_cur_it );
        pt_ref_it++;
        pt_cur_it++;
    }
}


Tracker::~Tracker()
{

}





    
}