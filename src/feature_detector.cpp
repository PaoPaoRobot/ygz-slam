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

#include <fast/fast.h>

#include "ygz/memory.h"
#include "ygz/feature_detector.h"
#include "ygz/config.h"
#include "ygz/frame.h"


namespace ygz {
    
// 构造：从Config中读取相关信息
FeatureDetector::FeatureDetector()
{
    _image_width = Config::get<int>("image.width");
    _image_height = Config::get<int>("image.height");
    _cell_size = Config::get<int>("feature.cell");
    _grid_rows = ceil(double(_image_height)/_cell_size);
    _grid_cols = ceil(double(_image_width)/_cell_size);
    _detection_threshold = Config::get<double>("feature.detection_threshold");
}

// 提取算法
void FeatureDetector::Detect(Frame::Ptr frame)
{
    // reset the feature grid
    frame->_grid = vector<int>(_grid_cols*_grid_rows, 0);

    Corners corners( _grid_cols*_grid_rows, Corner(0,0,_detection_threshold,0,0.0f));
    for(int L=0; L<frame->_pyramid_level; ++L)
    {
        const int scale = (1<<L);
        vector<fast::fast_xy> fast_corners;
#if __SSE2__
        fast::fast_corner_detect_10_sse2(
            (fast::fast_byte*) frame->_pyramid[L].data, frame->_pyramid[L].cols,
            frame->_pyramid[L].rows, frame->_pyramid[L].cols, 20, fast_corners);
#elif HAVE_FAST_NEON
        fast::fast_corner_detect_9_neon(
            (fast::fast_byte*) frame->_pyramid[L].data, frame->_pyramid[L].cols,
            frame->_pyramid[L].rows, frame->_pyramid[L].cols, 20, fast_corners);
#else
        fast::fast_corner_detect_10(
            (fast::fast_byte*) frame->_pyramid[L].data, frame->_pyramid[L].cols,
            frame->_pyramid[L].rows, frame->_pyramid[L].cols, 20, fast_corners);
#endif
        // nomax
        vector<int> scores, nm_corners;
        fast::fast_corner_score_10((fast::fast_byte*) frame->_pyramid[L].data, frame->_pyramid[L].cols, fast_corners, 20, scores);
        fast::fast_nonmax_3x3(fast_corners, scores, nm_corners);

        for(auto it=nm_corners.begin(), ite=nm_corners.end(); it!=ite; ++it)
        {
            fast::fast_xy& xy = fast_corners.at(*it);
            const int gy = static_cast<int>((xy.y*scale)/_cell_size);
            const int gx = static_cast<int>((xy.x*scale)/_cell_size);
            const size_t k = gy*_grid_cols+gx;
            if ( k > frame->_grid.size() ) {
                LOG(ERROR) << k <<" is larger than grid size "<<frame->_grid.size()<<endl;
                continue;
            }
            
            if( frame->_grid[k] == 1 )  // already have features here
                continue;
            const float score = this->ShiTomasiScore( frame->_pyramid[L], xy.x, xy.y );
            if(score > corners.at(k).score) {
                corners[k] = Corner(xy.x*scale, xy.y*scale, score, L, 0.0f);
                frame->_grid[k] = 1;
            }
        }
    }

    for ( Corner& c: corners ) {
        MapPoint point; 
        point._obs[frame->_id] = Vector3d( c.x, c.y, 1 );
        point._pyramid_level = c.level; 
        frame->_map_point_candidates.push_back( point );
    }

}

void FeatureDetector::SetExistingFeatures ( Frame::Ptr frame )
{
    frame->_grid = vector<int>(_grid_cols*_grid_rows, 0);
    for ( Vector3d& obs : frame->_observations ) {
        if ( obs[3] > 0 ) {
            // inlier observations 
            const int gy = static_cast<int>( obs[0]/_cell_size);
            const int gx = static_cast<int>( obs[1]/_cell_size);
            const size_t k = gy*_grid_cols+gx;
            frame->_grid[k] = 1;
        }
    }
}


float FeatureDetector::ShiTomasiScore(const cv::Mat& img, const int& u, const int& v) const
{
    assert(img.type() == CV_8UC1);

    float dXX = 0.0;
    float dYY = 0.0;
    float dXY = 0.0;
    const int halfbox_size = 4;
    const int box_size = 2*halfbox_size;
    const int box_area = box_size*box_size;
    const int x_min = u-halfbox_size;
    const int x_max = u+halfbox_size;
    const int y_min = v-halfbox_size;
    const int y_max = v+halfbox_size;

    if(x_min < 1 || x_max >= img.cols-1 || y_min < 1 || y_max >= img.rows-1)
        return 0.0; // patch is too close to the boundary

    const int stride = img.step.p[0];
    for( int y=y_min; y<y_max; ++y )
    {
        const uint8_t* ptr_left   = img.data + stride*y + x_min - 1;
        const uint8_t* ptr_right  = img.data + stride*y + x_min + 1;
        const uint8_t* ptr_top    = img.data + stride*(y-1) + x_min;
        const uint8_t* ptr_bottom = img.data + stride*(y+1) + x_min;
        for(int x = 0; x < box_size; ++x, ++ptr_left, ++ptr_right, ++ptr_top, ++ptr_bottom)
        {
            float dx = *ptr_right - *ptr_left;
            float dy = *ptr_bottom - *ptr_top;
            dXX += dx*dx;
            dYY += dy*dy;
            dXY += dx*dy;
        }
    }

    // Find and return smaller eigenvalue:
    dXX = dXX / (2.0 * box_area);
    dYY = dYY / (2.0 * box_area);
    dXY = dXY / (2.0 * box_area);
    return 0.5 * (dXX + dYY - sqrt( (dXX + dYY) * (dXX + dYY) - 4 * (dXX * dYY - dXY * dXY) ));
}


}
