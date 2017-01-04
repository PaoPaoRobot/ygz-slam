#include "ygz/common_include.h"
#include "ygz/ORB/ORBMatcher.h"
#include "ygz/frame.h"
#include "ygz/camera.h"

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

int ORBMatcher::CheckFrameDescriptors ( 
    Frame* frame1, 
    Frame* frame2, 
    vector<pair<int,int>>& matches
)
{
    vector<int> distance; 
    LOG(INFO) << frame1->_descriptors.size()<<","<<frame2->_descriptors.size();
    // 第一个帧有一些没跟上的点，所以会多一些
    
    // assert( frame1->_map_point_candidates.size() == frame2->_map_point_candidates.size() );
    // assert( frame1->_descriptors.size() == frame2->_descriptors.size() );
    
    vector<int> valid_idx_ref;
    for ( size_t i1 = 0, i2=0; i1<frame1->_map_point_candidates.size(); i1++ )
    {
        if ( frame1->_candidate_status[i1] == CandidateStatus::BAD )
            continue; 
        distance.push_back ( 
            DescriptorDistance( frame1->_descriptors[i1], frame2->_descriptors[i2] )
        );
        i2++;
        valid_idx_ref.push_back(i1);
    }
    
    int cnt_good = 0;
    int best_dist = *std::min_element( distance.begin(), distance.end() );
    LOG(INFO) << "best dist = "<<best_dist<<endl;
    
    // 取个上下限
    best_dist = best_dist>_options.init_low ? best_dist:_options.init_low; 
    best_dist = best_dist<_options.init_high ? best_dist:_options.init_high; 
    
    
    for ( size_t i=0; i<distance.size(); i++ ) {
        // LOG(INFO) << "dist = "<<distance[i]<<endl;
        
        if ( distance[i] < _options.initMatchRatio*best_dist )  {
            frame1->_candidate_status[ valid_idx_ref[i] ] = CandidateStatus::WAIT_TRIANGULATION;
            frame2->_candidate_status[i] = CandidateStatus::WAIT_TRIANGULATION;
            matches.push_back( make_pair( valid_idx_ref[i], i) );
            cnt_good++;
        }
        else  {
            frame1->_candidate_status[ valid_idx_ref[i] ] = CandidateStatus::BAD;
            frame2->_candidate_status[i] = CandidateStatus::BAD;
        }
    }
    
    LOG(INFO) << distance.size() <<","<<frame2->_candidate_status.size()<<endl;
    return cnt_good;
}

int ORBMatcher::SearchForTriangulation ( 
    Frame* kf1, 
    Frame* kf2, 
    const Matrix3d& E12, 
    vector< pair< int, int > >& matched_points, 
    const bool& onlyStereo )
{
    LOG(INFO) << kf2->_map_point_candidates.size()<<","<<kf2->_descriptors.size()<<endl;
    
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
                // 取出 kf1 中对应的特征点
                const cv::KeyPoint& kp1 = kf1->_map_point_candidates[idx1];
                const cv::Mat& desp1 = kf1->_descriptors[idx1];
                
                int bestDist = _options.th_low;
                int bestIdx2 = -1;
                
                for ( size_t i2=0, iend2 = f2it->second.size(); i2<iend2; i2++ ) {
                    size_t idx2 = f2it->second[i2];
                    Mat& desp2 = kf2->_descriptors[idx2];
                    const int dist = DescriptorDistance( desp1, desp2 );
                    
                    const cv::KeyPoint& kp2 = kf2->_map_point_candidates[idx2];
                    
                    
                    if ( dist>_options.th_low || dist>bestDist ) 
                        continue;
                    
                    LOG(INFO) << "dist = " << dist << endl;
                    
                    Mat ref_show = kf1->_color.clone();
                    Mat curr_show = kf2->_color.clone();
                    
                    cv::circle( ref_show, kp1.pt, 2, cv::Scalar(0,250,0), 2 );
                    cv::circle( curr_show, kp2.pt, 2, cv::Scalar(0,250,0), 2 );
                    cv::imshow("ref", ref_show);
                    cv::imshow("curr", curr_show);
                    cv::waitKey(0);
                    
                   
                    // 计算两个 keypoint 是否满足极线约束
                    if ( CheckDistEpipolarLine(kp1, kp2, F12) ) {
                        // 极线约束成立
                        bestIdx2 = idx2;
                        bestDist = dist;
                    }
                }
                
                if ( bestIdx2 >=0 ) {
                    const cv::KeyPoint& kp2 = kf2->_map_point_candidates[bestIdx2];
                    matches12[idx1] = bestIdx2;
                    matches++;
                    
                    if ( _options.checkOrientation ) {
                        float rot = kp1.angle - kf2->_map_point_candidates[bestIdx2].angle;
                        if ( rot<0 ) rot+=360;
                        int bin = round(rot*factor);
                        if ( bin == HISTO_LENGTH )
                            bin = 0;
                        assert( bin>=0 &&  bin<HISTO_LENGTH );
                        rotHist[bin].push_back( bestIdx2 );
                    }
                    
                }
                
                f1it++;
                f2it++;
            }
        } else if ( f1it->first < f2it->first ) {
            f1it = fv1.lower_bound( f2it->first );
        } else {
            f2it = fv2.lower_bound( f1it->first );
        }
    }
    
    if ( _options.checkOrientation ) {
        // TODO 去掉旋转不对的点
    }
    
    matched_points.clear();
    matched_points.reserve( matches );
    
    for ( size_t i=0; i<matches12.size(); i++ ) {
        if ( matches12[i] >= 0 )
            matched_points.push_back( make_pair(i, matches12[i]) );
    }
    
    LOG(INFO) << "matches: "<<matches;
    return matches;
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
                    if ( float(bestDist1) < _options.knnRatio*float(bestDist2) ) {
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

bool ORBMatcher::CheckDistEpipolarLine(const cv::KeyPoint& kp1, const cv::KeyPoint& kp2, const Matrix3d& F12)
{
    const float a = kp1.pt.x * F12(0,0) + kp1.pt.y*F12(1,0)+F12(2,0);
    const float b = kp1.pt.x * F12(0,1) + kp1.pt.y*F12(1,1)+F12(2,1);
    const float c = kp1.pt.x * F12(0,2) + kp1.pt.y*F12(1,2)+F12(2,2);
    
    const float num = a*kp2.pt.x+b*kp2.pt.y+c;
    const float den = a*a+b*b;
    
    LOG(INFO) << "den = "<<den;
    if ( den < 1e-6 )
        return false; 
    
    const float dsqr = num*num/den;
    LOG(INFO) << "dsqr = "<<dsqr << endl;
    return dsqr < 3.84 * (1<<kp2.octave);
}



    
}