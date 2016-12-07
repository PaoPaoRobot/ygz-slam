#include "ygz/frame.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace ygz {
    
void Frame::InitFrame()
{
    // convert the color into CV_32F grayscale 
    Mat gray; 
    cv::cvtColor( _color, gray, CV_BGR2GRAY );
    _pyramid_level = Config::get<int>("frame.pyramid");
    _pyramid.resize( _pyramid_level );
    _pyramid[0] = gray;
    CreateImagePyramid();
}

void Frame::CreateImagePyramid()
{
    for ( size_t i=1; i<_pyramid.size(); i++ ) {
        // 在CV里使用down构造分辨率更低的图像，中间有高斯模糊化以降低噪声
        cv::pyrDown( _pyramid[i-1], _pyramid[i] ); 
    }
}

PinholeCamera::Ptr Frame::_camera = nullptr;
}
