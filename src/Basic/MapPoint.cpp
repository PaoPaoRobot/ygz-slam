#include "ygz/Basic/MapPoint.h"
#include "ygz/Basic/Memory.h"
#include "ygz/Basic/Camera.h"
// #include "ygz/ORB/ORBMatcher.h"

namespace ygz 
{
    
    /*
void MapPoint::ComputeDistinctiveDesc()
{
    if ( _observations.empty() ) return; 
    // 两两的距离
    vector<vector<int>> distances; 
    distances.resize( _observations.size(), vector<int>( _observations.size(),0) );
    
    int i=0; 
    for ( auto iter1=_observations.begin(); 
         iter1!=_observations.end(); iter1++,i++ )  
    {
        distances[i][i] = 0;
        int j = i+1;
        for ( auto iter2=iter1+1; iter2!=_observations.end(); iter2++,j++ ) 
        {
            distances[i][j] = ORBMatcher::DescriptorDistance( iter1->_desc[i], iter2->_desc[j] );
            distances[j][i] = distances[i][j];
        }
    }
    
    int bestMidian = INT_MAX;
    auto bestIdx = _observations.begin();
   
    int i=0;
    for ( auto iter=_observations.begin(); 
         iter!=_observations.end(); iter++,i++ ) {
        
        vector<int> dists( distances[i].begin(), distances[i].end() );
        sort( dists.begin(), dists.end() );
        int median = dists[0.5*_observations.size()-1];
        if ( median < bestMidian ) {
            bestMidian = median;
            bestIdx = iter;
        }
    }
    
    _distinctive_desc = bestIdx->_desc.clone();
}*/

}
