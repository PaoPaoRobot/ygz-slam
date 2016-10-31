#include "ygz/config.h"
#include "ygz/system.h"
#include "ygz/visual_odometry.h"
#include "ygz/local_mapping.h"
#include "ygz/loop_closing.h"


namespace ygz
{

bool System::Initialize( const string& config_file )
{
    LOG(INFO) << "Starting SLAM system ..."<<endl;
    bool ret = Config::setParameterFile( config_file );
    if ( !ret )
    {
        return false;
    }

    string sensor_type = Config::get<string>("system.sensor");
    if ( sensor_type == "monocular" ) {
        _sensor_type = MONOCULAR;
    }
    else if (sensor_type == "stereo") {
        _sensor_type = STEREO;
    }
    else if ( sensor_type == "RGBD") {
        _sensor_type = RGBD;
    }
    else {
        LOG(ERROR) << "Unrecognized sensor type, system initialization failed! "<<endl;
        return false; 
    }
    
    // init visual odometry 
    string vo_type = Config::get<string>("system.vo");
    if ( vo_type == "sparse_direct") {
        _vo_type = SPARSE_DIRECT;
    } else if ( vo_type == "sparse_orb") {
        _vo_type = SPARSE_ORB;
    } else if ( vo_type == "semi_dense_direct" ) {
        _vo_type = SEMI_DENSE_DIRECT;
    } else {
        LOG(ERROR) << "Unrecognized vo type, system initialization failed! " << endl; 
        return false; 
    }
    
    // init mapping 
    string map_type = Config::get<string>("system.map");
    if ( map_type == "SPARSE" ) {
        _map_type = SPARSE;
    } else if ( map_type == "SEMI_DENSE" ) {
        _map_type = SEMI_DENSE; 
    } else if ( map_type == "DENSE" ) {
        _map_type = DENSE;
    } else {
        LOG(ERROR) << "Unrecognized map type, system initialization failed! " << endl; 
        return false; 
    }
    
    if ( _map_type == DENSE && _sensor_type != RGBD ) {
        LOG(ERROR) << "Dense map is only runnable in RGBD mode, system intialization failed! " << endl; 
        return false;
    }
        
    // initialize the system 
    _visual_odometry = new VisualOdometry();
    
    if ( Config::get<string>("system.localmapping") == "yes" ) {
        _local_mapping = new LocalMapping();
    } else {
        LOG(INFO) << "Local Mapping disabled, performance maybe not good." << endl; 
    }
    
    if ( Config::get<string>("system.loopclosing") == "yes" ) {
        _loop_closing = new LoopClosing();
    } else {
        LOG(INFO) << "Loop closing disabled, performance maybe not good." << endl; 
    }
    
    
    return true; 
}

SE3 System::TrackMonocular( const Mat& image, const double& timestamp ) {
    return SE3(); 
}

SE3 System::TrackStereo( const Mat& image_left, const Mat& image_right, const double& timestamp ) {
    return SE3(); 
}

SE3 System::TrackRGBD( const Mat& color, const Mat& depth, const double& timestamp ) {
    return SE3();
}

void System::Reset() {
    LOG(INFO) << "Resetting system. " << endl;
}

void System::Shutdown() {
    LOG(INFO) << "Shutting down the system. " << endl; 
}


}
