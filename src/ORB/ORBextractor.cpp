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
        kp.angle = ygz::IC_Angle(image, kp.pt, umax);
    }
    
    _orb->compute( image, keypoints, descriptors );
}
    
void ORBExtractor::Compute ( 
    const Mat& image, const Vector2d& corners, 
    cv::KeyPoint& keypoints, Mat& descriptors 
)
{
    keypoints.pt.x = corners[0];
    keypoints.pt.y = corners[1];
    keypoints.octave = 1; 
    keypoints.angle = ygz::IC_Angle( image, keypoints.pt, umax );
    vector<cv::KeyPoint> kp{keypoints};
    _orb->compute( image, kp, descriptors );

}

float IC_Angle(
    const Mat& image, cv::Point2f pt,  
    const vector<int> & u_max
)
{
    int m_01 = 0, m_10 = 0;
    const uchar* center = &image.at<uchar> (cvRound(pt.y), cvRound(pt.x));

    // Treat the center line differently, v=0
    for (int u = -HALF_PATCH_SIZE; u <= HALF_PATCH_SIZE; ++u)
        m_10 += u * center[u];

    // Go line by line in the circuI853lar patch
    int step = (int)image.step1();
    for (int v = 1; v <= HALF_PATCH_SIZE; ++v)
    {
        // Proceed over the two lines
        int v_sum = 0;
        int d = u_max[v];
        for (int u = -d; u <= d; ++u)
        {
            int val_plus = center[u + v*step], val_minus = center[u - v*step];
            v_sum += (val_plus - val_minus);
            m_10 += u * (val_plus + val_minus);
        }
        m_01 += v * v_sum;
    }

    return cv::fastAtan2((float)m_01, (float)m_10);
}
    
}