#include "ygz/map_point.h"
#include "ygz/memory.h"
#include "ygz/camera.h"
#include "ygz/ORB/ORBMatcher.h"

namespace ygz 
{
    
Vector3d MapPoint::GetObservedPt ( const long unsigned int& keyframe_id ) 
{
    auto p = _obs[keyframe_id];
    return Memory::GetFrame(keyframe_id)->_camera->Pixel2Camera( p.head<2>(), p[2] );
}

void MapPoint::ComputeDistinctiveDesc()
{
    if ( _descriptors.empty() ) return; 
    // 两两的距离
    vector<vector<int>> distances; 
    distances.resize( _descriptors.size(), vector<int>(_descriptors.size(),0) );
    for ( size_t i=0; i<_descriptors.size(); i++ )  {
        distances[i][i] = 0;
        for ( size_t j=i+1; j<_descriptors.size(); j++ ) {
            distances[i][j] = ORBMatcher::DescriptorDistance( _descriptors[i], _descriptors[j] );
            distances[j][i] = distances[i][j];
        }
    }
    
    int bestMidian = INT_MAX;
    int bestIdx = 0;
    for ( size_t i=0; i<_descriptors.size(); i++ ) {
        vector<int> dists( distances[i].begin(), distances[i].end() );
        sort( dists.begin(), dists.end() );
        int median = dists[0.5*_descriptors.size()-1];
        if ( median < bestMidian ) {
            bestMidian = median;
            bestIdx = i;
        }
    }
    
    _distinctive_desc = _descriptors[bestIdx].clone();
}

}
