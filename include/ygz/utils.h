#ifndef YGZ_UTILS_H
#define YGZ_UTILS_H

#include "ygz/common_include.h"

// 一些杂七杂八不知道放哪的东西 
namespace ygz {
    
namespace utils {
    
// 转换函数 
inline Eigen::Vector2d Cv2Eigen( const cv::Point2f& p ) {
    return Eigen::Vector2d( p.x, p.y );
}
    
}
}


#endif