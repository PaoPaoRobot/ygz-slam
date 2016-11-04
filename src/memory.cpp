#include "ygz/memory.h"

namespace ygz {
    
void Memory::clean()
{
    _frames.clear();
    _points.clear();
}

Frame::Ptr Memory::CreateNewFrame()
{
    
}


shared_ptr<Memory> Memory::_mem = nullptr;

}