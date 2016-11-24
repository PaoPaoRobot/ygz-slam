#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

#include "ygz/common_include.h"
#include "ygz/g2o_types.h"

namespace ygz
{

namespace opti
{

// two view BA, used in intialization 
void TwoViewBA (
    const unsigned long& frameID1,
    const unsigned long& frameID2
);

}
}

#endif
