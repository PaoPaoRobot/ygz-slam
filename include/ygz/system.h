#ifndef SYSTEM_H_
#define SYSTEM_H_

#include "ygz/common_include.h"

namespace ygz 
{
// forward declare 
class VisualOdometry;
class Map; 
class LocalMapping; 
class LoopClosing;
    
class System 
{
public:
    // all these types can be configured through config file 
    // which sensor 
    enum SensorType {
        MONOCULAR = 0,
        STEREO, 
        RGBD, 
    }; 
    
    // which VO do you want 
    enum VOType {
        SPARSE_DIRECT,      // sparse direct method, like SVO 
        SPARSE_ORB,         // sparse feature method, like ORB
        SEMI_DENSE_DIRECT,  // semi-dense method, like LSD
    };
    
    // which map do you want
    enum MapType {
        SPARSE = 0, 
        SEMI_DENSE = 1,
        DENSE = 2  // only in RGBD mode 
        // TODO add octomaps and surfels here 
    };
    
    
public:
    System() {} 
    
    // initialize the system, given the config file
    bool Initialize( const string& config_file );
    
    // Track monocular images, return T_cw of this image 
    SE3 TrackMonocular( const Mat& image, const double & timestamp );  
    
    // Track stereo images, return T_cw of this image 
    SE3 TrackStereo( const Mat& image_left, const Mat& image_right, const double& timestamp );
    
    // Track RGBD images, return T_cw of this image 
    SE3 TrackRGBD( const Mat& color, const Mat& depth, const double& timestamp ); 
    
    // reset 
    void Reset(); 
    
    // shutdown 
    void Shutdown(); 
    
    // save outputs 
    void SaveTrajectory( const string& filename );
    void SaveMap( const string& filename ); 
    
    // load outputs 
    void LoadMap( const string& filename, const MapType& map_type );
    
    // accessors 
    SensorType GetSensorType() const { return  _sensor_type; }
    MapType GetMapType() const { return  _map_type; }
    VOType GetVOType() const { return  _vo_type; }
    
protected:
    SensorType  _sensor_type=MONOCULAR;
    MapType     _map_type=SPARSE;
    VOType      _vo_type=SPARSE_DIRECT; 
    
    
    // three main threads like in ORB_SLAM
    VisualOdometry* _visual_odometry =nullptr; 
    LocalMapping* _local_mapping =nullptr; 
    LoopClosing* _loop_closing =nullptr; 
    
    // TODO add thread safe things 
};


}

#endif
