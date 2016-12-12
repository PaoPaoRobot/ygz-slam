/*
 * Copyright (c) 2016 <copyright holder> <email>
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 */

#ifndef FEATUREDETECTOR_H_
#define FEATUREDETECTOR_H_

#include "ygz/common_include.h"

namespace ygz
{
    
// 角点，在提取FAST中用到
struct Corner {
    int x=0;        //!< x-coordinate of corner in the image.
    int y=0;        //!< y-coordinate of corner in the image.
    int level=0;    //!< pyramid level of the corner.
    float score=0;  //!< shi-tomasi score of the corner.
    float angle=0;  //!< for gradient-features: dominant gradient angle.
    Corner ( int x, int y, float score, int level, float angle ) :
        x ( x ), y ( y ), level ( level ), score ( score ), angle ( angle )
    {}
    
    Corner() {}
};

typedef vector<Corner> Corners;

// 特征提取算法
// 默认提取网格化的FAST特征，提取完成后存储在frame->_map_point_candidates中，
// 但是这些点没有深度，需要等到深度滤波器收敛才有有效的值
// 在Tracker和visual odometry中均有使用
class FeatureDetector
{
private:
    // params
    int _image_width=640, _image_height=480;    // 图像长宽，用以计算网格
    int _cell_size;                             // 网格大小
    int _grid_rows=0, _grid_cols=0;             // 网格矩阵的行和列
    double _detection_threshold =20.0;          // 特征响应阈值
    
public:
    FeatureDetector();
    
    // 提取一个帧中的特征点，记录于frame::_map_point_candidates
    // 参数可以指定是否覆盖之前的特征点
    void Detect ( Frame::Ptr frame, bool overwrite_existing_features=true );
    
    // 设置已经存在的特征点
    void SetExistingFeatures( Frame::Ptr frame );

private:
    // Shi-Tomasi 分数，这个分数越高则特征越优先
    float ShiTomasiScore ( const Mat& img, const int& u, const int& v ) const ;


};
}

#endif // FEATUREDETECTOR_H_
