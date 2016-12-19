add_library( ygz-slam SHARED
    frame.cpp
    memory.cpp
    tracker.cpp
    config.cpp 
    map_point.cpp
    system.cpp
    visual_odometry.cpp
    feature_detector.cpp
    initializer.cpp
    optimizer.cpp
    local_mapping.cpp
    utils.cpp
    viewer.cpp
    
    ORB/ORBextractor.cpp 
)
