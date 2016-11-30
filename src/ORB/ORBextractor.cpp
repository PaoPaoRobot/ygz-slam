#include "ygz/ORB/ORBextractor.h"

namespace ygz {
    
ORBExtractor::ORBExtractor() 
{
    umax.resize(HALF_PATCH_SIZE + 1);
    int v, v0, vmax = cvFloor(HALF_PATCH_SIZE * sqrt(2.f) / 2 + 1);
    int vmin = cvCeil(HALF_PATCH_SIZE * sqrt(2.f) / 2);
    for (v = HALF_PATCH_SIZE, v0 = 0; v >= vmin; --v)
    {
        while (umax[v0] == umax[v0 + 1])
            ++v0;
        umax[v] = v0;
        ++v0;
    }
    
    _orb = cv::ORB::create();
}
    
    
void  ORBExtractor::Compute ( 
    const cv::Mat& image, 
    const vector<Vector2d>& corners, 
    vector<cv::KeyPoint>& keypoints, 
    cv::Mat& descriptors 
) 
{
    keypoints.resize( corners.size() );
    for ( int i=0; i<corners.size(); i++ ) {
        const Vector2d& corner = corners[i];
        cv::KeyPoint kp;
        kp.pt.x = corner[0];
        kp.pt.y = corner[1];
        kp.octave = 1; 
        // the orientation 
        kp.angle = IC_Angle(image, kp.pt, umax);
    }
    
    _orb->compute( image, keypoints, descriptors );
}
    
    
}