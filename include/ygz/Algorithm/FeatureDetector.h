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

#ifndef YGZ_FEATUREDETECTOR_H_
#define YGZ_FEATUREDETECTOR_H_

#include "ygz/Basic/Common.h"
#include "ygz/Basic/Frame.h"

namespace ygz
{

// 特征提取算法
// 默认提取网格化的FAST特征，完成后计算旋转和描述子
// 利用SSE加速后,要比OpenCV的快一些,但是特征数量无法保证
// 特征点刚提出时没有深度，进行三角化后才会有深度
// 在 Tracker 和 Visual Odometry中均有使用

class FeatureDetector
{
public:
    // ORB 相关常量
    const int PATCH_SIZE = 31;
    const int HALF_PATCH_SIZE = 15;
    const int EDGE_THRESHOLD = 19;
    
    struct Option {
        int _image_width=640, _image_height=480;    // 图像长宽，用以计算网格
        int _cell_size;                             // 网格大小
        int _grid_rows=0, _grid_cols=0;             // 网格矩阵的行和列
        double _detection_threshold =20.0;          // 特征响应阈值
    } _option;
    
    FeatureDetector();
    
    // 从参数文件中读取参数(否则使用option中默认的参数)
    void LoadParams(); 

    // 提取一个帧中的特征点，记录于 frame->_features 中,同时会计算描述
    void Detect ( Frame* frame, bool overwrite_existing_features=true );
    
    // 计算frame中关键点的旋转和描述子
    // 这种情况出现在初始化追踪完成时。由于光流只能追踪特征点的图像坐标，所以从初始化的第一个帧到第二个帧时，需要把
    // 第二个帧的像素点转化为带有特征描述的特征点
    void ComputeAngleAndDescriptor( Frame* frame );
    
    void ComputeDescriptor( Feature* fea );

private:
    // 设置已有特征的网格
    void SetExistingFeatures ( Frame* frame );
    
    // 计算 FAST 角度
    float IC_Angle(
        const Mat& image, 
        const Vector2d& pt,  
        const vector<int> & u_max
    );
    
    void ComputeOrbDescriptor(
        const Feature* feature,
        const Mat& img, 
        const cv::Point* pattern,
        uchar* desc
    );
    
    // Shi-Tomasi 分数，这个分数越高则特征越优先
    float ShiTomasiScore ( const Mat& img, const int& u, const int& v ) const ;
    
    vector<Feature*> _old_features;
    vector<Feature*> _new_features;
    
    // 计算ORB 描述时需要用的常量
    std::vector<int>    _umax;
    vector<cv::Point>   _pattern;
};
}

#endif // FEATUREDETECTOR_H_
