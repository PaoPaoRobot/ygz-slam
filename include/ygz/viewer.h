#ifndef YGZ_VIEWER_H_
#define YGZ_VIEWER_H_

#include <pangolin/pangolin.h>
#include "ygz/memory.h"

namespace ygz {
    
class Viewer {
friend class Memory; 
    
public:
    Viewer(); 
    
    void Draw(); 
    
protected:
    pangolin::View _dcam; 
    pangolin::OpenGlRenderState _scam;
};
    
}


#endif 