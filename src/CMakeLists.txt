# core 
add_library( ygz-core SHARED
    Basic/Config.cpp
    Basic/Frame.cpp
    Basic/MapPoint.cpp
    Basic/Memory.cpp
)

# algorithm
add_library( ygz-algorithm SHARED
    Algorithm/FeatureDetector.cpp 
    Algorithm/Matcher.cpp 
    Algorithm/CVUtils.cpp
    Algorithm/Tracker.cpp
    Algorithm/Initializer.cpp
    Algorithm/BA.cpp
)

target_link_libraries( ygz-algorithm 
    ygz-core ${THIRD_PARTY_LIBS}
)
    

# modules
add_library( ygz-module SHARED 
  Module/VisualOdometry.cpp
  Module/LocalMapping.cpp
)

target_link_libraries( ygz-module
    ygz-core ygz-algorithm ${THIRD_PARTY_LIBS}
)
