#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "ygz/frame.h"
#include "ygz/memory.h"

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

bool Frame::GetMeanAndMinDepth ( double& mean_depth, double& min_depth )
{
    mean_depth = 0; min_depth = 9999;
    int cnt_valid_points = 0;
    for ( unsigned long& map_point_id : _map_point ) {
        MapPoint::Ptr map_point = Memory::GetMapPoint( map_point_id );
        if ( map_point->_bad ) 
            continue;
        double depth = this->_camera->World2Camera( map_point->_pos_world, this->_T_c_w )[2];
        if ( depth<0 ) 
            continue; 
        LOG(INFO) << "depth = "<<depth<<endl;
        if ( depth>10 || depth < 0.05 ) {
            // 明显不对啊
            LOG(INFO) << map_point->_pos_world.transpose() << endl;
        }
        
        cnt_valid_points ++ ;
        mean_depth += depth;
        if ( depth < min_depth ) 
            min_depth = depth;
    }
    
    if ( cnt_valid_points == 0 ) {
        mean_depth = 0; 
        min_depth = 0;
        return false; 
    }
    
    mean_depth /= cnt_valid_points;
    return true; 
}


PinholeCamera::Ptr Frame::_camera = nullptr;

}
