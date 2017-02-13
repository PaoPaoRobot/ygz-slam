add_library( ygz-core SHARED
    Basic/Config.cpp
    Basic/Frame.cpp
    Basic/MapPoint.cpp
    Basic/Memory.cpp
)

add_library( ygz-algorithm 
    Algorithm/FeatureDetector.cpp 
    Algorithm/Matcher.cpp 
    Algorithm/CVUtils.cpp
    Algorithm/Tracker.cpp
)