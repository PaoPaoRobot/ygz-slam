#include "ygz/map_point.h"
#include "ygz/memory.h"
#include "ygz/camera.h"

namespace ygz 
{
    
Vector3d MapPoint::GetObservedPt ( const long unsigned int& keyframe_id ) 
{
    auto p = _obs[keyframe_id];
    return Memory::GetFrame(keyframe_id)->_camera->Pixel2Camera( p.head<2>(), p[2] );
}


}
