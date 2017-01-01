#include "ygz/ORB/ORBMatcher.h"
#include "ygz/frame.h"
#include <ygz/camera.h>

namespace ygz {
    
ORBMatcher::ORBMatcher ()
{
    _options.th_low = Config::get<int>("matcher.th_low");
    _options.th_high = Config::get<int>("matcher.th_high");
    _options.knnRatio = Config::get<int>("matcher.knnRatio");
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
        if ( distance[i] < _options.knnRatio*best_dist ) 
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
    Vector2d c2_px = kf2->_camera->World2Pixel( kf1->GetCamCenter(), kf2->_T_c_w );

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

// 利用 Bag of Words 加速匹配
int ORBMatcher::SearchByBoW(Frame* kf1, Frame* kf2, map<int,int>& matches)
{
    DBoW3::FeatureVector& fv1 = kf1->_feature_vec;
    DBoW3::FeatureVector& fv2 = kf2->_feature_vec;
    
    int cnt_matches = 0;
    
    vector<int> rotHist[HISTO_LENGTH]; // rotation 的统计直方图
    for ( int i=0;i <HISTO_LENGTH; i++ )
        rotHist[i].reserve(500);
    float factor = 1.0f/HISTO_LENGTH;
    
    DBoW3::FeatureVector::const_iterator f1it = fv1.begin();
    DBoW3::FeatureVector::const_iterator f2it = fv2.begin();
    DBoW3::FeatureVector::const_iterator f1end = fv1.end();
    DBoW3::FeatureVector::const_iterator f2end = fv2.end();
       
    while( f1it!=f1end && f2it!=f2end ) {
        if ( f1it->first == f2it->first ) {
            const vector<unsigned int> indices_f1 = f1it->second;
            const vector<unsigned int> indices_f2 = f2it->second;
            
            // 遍历 f1 中该 node 的特征点
            for ( size_t if1 = 0; if1<indices_f1.size(); if1++ ) {
                const unsigned int real_idx_f1 = indices_f1[if1];
                Mat desp_f1 = kf1->_descriptors[real_idx_f1];
                int bestDist1 = 256;  // 最好的距离
                int bestIdxF2 = -1;
                int bestDist2 = 256;  // 第二好的距离
                
                for ( size_t if2=0; if2<indices_f2.size(); if2++) {
                    const unsigned int real_idx_f2 = indices_f2[if2];
                    const Mat& desp_f2 = kf2->_descriptors[real_idx_f2];
                    const int dist = DescriptorDistance( desp_f1, desp_f2 );
                    if ( dist < bestDist1 ) {
                        bestDist2 = bestDist1;
                        bestDist1 = dist;
                        bestIdxF2 = real_idx_f2;
                    } else if ( dist<bestDist2 ) {
                        bestDist2 = dist; 
                    }
                }
                
                if ( bestDist1 <= _options.th_low ) {
                    // 最小匹配距离小于阈值
                    if ( float(bestDist1) < _options.knnRatio* float(bestDist2) ) {
                        // 最好的匹配明显比第二好的匹配好
                        matches[ real_idx_f1 ] = bestIdxF2;
                        if ( _options.checkOrientation ) {
                            cv::KeyPoint& kp = kf1->_map_point_candidates[real_idx_f1];
                            float rot = kp.angle - kf2->_map_point_candidates[bestIdxF2].angle;
                            if ( rot<0 ) rot+=360;
                            int bin = round(rot*factor);
                            if ( bin == HISTO_LENGTH )
                                bin = 0;
                            assert( bin>=0 &&  bin<HISTO_LENGTH );
                            rotHist[bin].push_back( bestIdxF2 );
                        }
                        cnt_matches++;
                    }
                }
            }
            
            f1it++;
            f2it++;
            
        } else if ( f1it->first < f2it->first ) {       // f1 iterator 比较小
            f1it = fv1.lower_bound( f2it->first );
        } else {        // f2 iterator 比较少
            f2it = fv2.lower_bound( f1it->first );
        }
    }
        
    if ( _options.checkOrientation ) {
        // 根据方向删除误匹配
        int ind1 = -1; 
        int ind2 = -1;
        int ind3 = -1;
        
        ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3 );
        
        for ( int i=0; i<HISTO_LENGTH; i++ ) {
            if ( i==ind1 || i==ind2 || i==ind3 )  // 保留之
                continue;
            for ( size_t j=0; j<rotHist[i].size(); j++ ) {
                rotHist[i][j];
                // TODO 删掉值为 rotHist[i][j] 的匹配
                
                cnt_matches--;
            }
        }
    }
    
    return cnt_matches;
}


void ORBMatcher::ComputeThreeMaxima(vector<int>* histo, const int L, int& ind1, int& ind2, int& ind3)
{
    int max1=0;
    int max2=0;
    int max3=0;

    for(int i=0; i<L; i++)
    {
        const int s = histo[i].size();
        if(s>max1)
        {
            max3=max2;
            max2=max1;
            max1=s;
            ind3=ind2;
            ind2=ind1;
            ind1=i;
        }
        else if(s>max2)
        {
            max3=max2;
            max2=s;
            ind3=ind2;
            ind2=i;
        }
        else if(s>max3)
        {
            max3=s;
            ind3=i;
        }
    }

    if(max2<0.1f*(float)max1)
    {
        ind2=-1;
        ind3=-1;
    }
    else if(max3<0.1f*(float)max1)
    {
        ind3=-1;
    }
}


    
}