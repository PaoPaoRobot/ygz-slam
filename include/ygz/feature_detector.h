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
#include "ygz/frame.h"

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
    cv::Ptr<cv::FastFeatureDetector>  _detector;// OpencCV's FAST corner

public:
    FeatureDetector();

    // 提取一个帧中的特征点，记录于frame::_map_point_candidates
    // 参数可以指定是否覆盖之前的特征点
    void Detect ( Frame* frame, bool overwrite_existing_features=true ) {
        // 这东西偶尔会挂掉，原因不明，试试改用opencv的fast？
        LOG ( INFO ) <<"step into detect"<<endl;
        if ( overwrite_existing_features ) {
            frame->_grid = vector<int> ( _grid_cols*_grid_rows, 0 );
            frame->_map_point_candidates.clear();
        }

        vector<cv::KeyPoint> selected_kps;
        try {
            selected_kps.resize ( _grid_cols*_grid_rows );
        } catch (...) {
            LOG(FATAL) << "resize failed. "<<endl; 
        }

        LOG ( INFO ) <<"step into loop"<<endl;
        for ( size_t l=0; l<frame->_pyramid_level; l++ ) {
            vector<cv::KeyPoint> kps;
            kps.reserve ( 3000 );
            int scale = ( 1<<l );
            _detector->detect ( frame->_pyramid[l], kps );

            LOG ( INFO ) << "keypoints in level "<<l<<" : "<<kps.size() <<endl;
            for ( cv::KeyPoint& kp: kps ) {
                const int gy = static_cast<int> ( ( kp.pt.y*scale ) /_cell_size );
                const int gx = static_cast<int> ( ( kp.pt.x*scale ) /_cell_size );
                const size_t k = gy*_grid_cols+gx;
                if ( k > frame->_grid.size() ) {
                    LOG ( ERROR ) << k <<" is larger than grid size "<<frame->_grid.size() <<endl;
                    continue;
                }

                // non-overwrite, and already have features here
                if ( overwrite_existing_features==false && frame->_grid[k] == 1 ) {
                    continue;
                }
                if ( kp.response > selected_kps[k].response ) {
                    kp.octave = l;
                    kp.pt.x *= scale;
                    kp.pt.y *= scale;
                    selected_kps[k] = kp;
                    frame->_grid[k] = 1;
                }
            }
        }

        for ( cv::KeyPoint& kp: selected_kps ) {
            if ( frame->InFrame ( kp.pt ) == false ) {
                continue;
            }
            MapPoint point;
            point._first_observed_frame = frame->_id;
            point._obs[frame->_id] = Vector3d ( kp.pt.x, kp.pt.y, 1 );
            point._pyramid_level = kp.octave;
            frame->_map_point_candidates.push_back ( point );
        }

        LOG ( INFO ) << "add total "<<frame->_map_point_candidates.size() <<" new features. "<<endl;
        selected_kps.clear();
    }

    // 设置已经存在的特征点，这里根据frame->_observation来设的
    void SetExistingFeatures ( Frame* frame );

private:
    // Shi-Tomasi 分数，这个分数越高则特征越优先
    float ShiTomasiScore ( const Mat& img, const int& u, const int& v ) const ;


};
}

#endif // FEATUREDETECTOR_H_
