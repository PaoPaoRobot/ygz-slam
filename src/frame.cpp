#include "ygz/frame.h"
#include <opencv2/imgproc/imgproc.hpp>

namespace ygz {
    
void Frame::InitFrame()
{
    // convert the color into CV_32F grayscale 
    Mat gray; 
    cv::cvtColor( _color, gray, CV_BGR2GRAY );
    _pyramid.resize( _pyramid_level );
    // gray.convertTo( _pyramid[0], CV_32F );
    _pyramid[0] = gray;
    CreateImagePyramid();
}

void Frame::CreateImagePyramid()
{
    for ( size_t i=1; i<_pyramid.size(); i++ ) {
        _pyramid[i] = Mat( _pyramid[i-1].rows/2, _pyramid[i-1].cols/2, CV_32F );
        cv::pyrUp( _pyramid[i-1], _pyramid[i], cv::Size( _pyramid[i].rows, _pyramid[i].cols ));
    }
}

PinholeCamera::Ptr Frame::_camera = nullptr;
int Frame::_pyramid_level =3;
}
