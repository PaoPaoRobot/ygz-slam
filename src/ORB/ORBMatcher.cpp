#include "ygz/ORB/ORBMatcher.h"
#include "ygz/frame.h"
#include <ygz/camera.h>

namespace ygz {
    
ORBMatcher::ORBMatcher ( float nnratio, bool checkOri ): _nnRatio(nnratio), _checkOrientation( checkOri )
{

}

int ORBMatcher::DescriptorDistance ( const Mat& a, const Mat& b )
{
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();
    int dist=0;
    for(int i=0; i<8; i++, pa++, pb++)
    {
        unsigned  int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }
    return dist;
}

bool ORBMatcher::CheckFrameDescriptors ( 
    Frame* frame1, 
    Frame* frame2, 
    vector< bool >& inliers )
{
    vector<int> distance; 
    assert( frame1->_map_point_candidates.size() == frame2->_map_point_candidates.size() );
    for ( auto iter1 = frame1->_descriptors.begin(), iter2=frame2->_descriptors.begin(); 
        iter1!=frame1->_descriptors.end(); iter1++, iter2++
    ) {
        distance.push_back( 
            DescriptorDistance( *iter1, *iter2 )
        );
    }
    
    int best_dist = *std::min_element( distance.begin(), distance.end() );
    LOG(INFO) << "best dist = "<<best_dist<<endl;
    // 取个上下限
    best_dist = best_dist>_options.th_low ? best_dist:_options.th_low; 
    best_dist = best_dist<_options.th_high ? best_dist:_options.th_high; 
    for ( size_t i=0; i<distance.size(); i++ ) {
        // LOG(INFO) << "dist = "<<distance[i]<<endl;
        if ( distance[i] < _options.ratio*best_dist ) 
            inliers[i] = true;
        else 
            inliers[i] = false;
    }
}

int ORBMatcher::SearchForTriangulation ( 
    Frame* kf1, 
    Frame* kf2, 
    const Matrix3d& F12, 
    vector< pair< size_t, size_t > >& matched_points, 
    const bool& onlyStereo )
{
    DBoW3::FeatureVector& fv1 = kf1->_feature_vec;
    DBoW3::FeatureVector& fv2 = kf2->_feature_vec;
    
    // 极点像素坐标
    Vector3d c2_px = kf2->_camera->World2Pixel( kf1->GetCamCenter(), kf2->_T_c_w );

    // 计算匹配
    int matches = 0;
    vector<bool> matched2( kf2->_map_point_candidates.size(), false ); 
    vector<int> matches12( kf1->_map_point_candidates.size(), -1 );
    vector<int> rotHist[ HISTO_LENGTH ];
    for ( int i=0; i<HISTO_LENGTH; i++ ) {
        rotHist[i].reserve(500);
    }
    const float factor = 1.0f/HISTO_LENGTH;
    
    // 将属于同一层的ORB进行匹配，利用字典加速
    DBoW3::FeatureVector::const_iterator f1it = fv1.begin();
    DBoW3::FeatureVector::const_iterator f2it = fv2.begin();
    DBoW3::FeatureVector::const_iterator f1end = fv1.end();
    DBoW3::FeatureVector::const_iterator f2end = fv2.end();
    
    while( f1it!=f1end && f2it!=f2end ) {
        if ( f1it->first == f2it->first ) {
            // 同属一个节点
            for ( size_t i1=0; i1<f1it->second.size(); i1++ ) {
                const size_t idx1 = f1it->second[i1]; 
                // 这里略绕，稍微放一放，等我把特征点的描述搞定
            }
        } else if ( f1it->first < f2it->first ) {
            f1it = fv1.lower_bound( f2it->first );
        } else {
            f2it = fv2.lower_bound( f1it->first );
        }
    }
}



    
}