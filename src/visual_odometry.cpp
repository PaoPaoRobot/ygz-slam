#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "ygz/visual_odometry.h"
#include "ygz/optimizer.h"
#include "ygz/memory.h"

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
        // initialized, do tracking
        if ( _status == VO_GOOD ) {
            // track ref or velocity
            TrackRefFrame();
            
            
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
            SetKeyframe( _ref_frame );
            SetKeyframe( _curr_frame );
            
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
            cv::waitKey(0);
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
}

void VisualOdometry::TrackRefFrame()
{
    opti::SparseImgAlign align;
    align.SparseImageAlignmentCeres( _ref_frame, _curr_frame, 0 );
}


}
