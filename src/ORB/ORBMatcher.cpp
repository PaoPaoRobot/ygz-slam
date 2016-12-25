#include "ygz/ORB/ORBMatcher.h"
#include "ygz/frame.h"

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



    
}