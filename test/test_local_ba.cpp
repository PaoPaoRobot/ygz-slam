#include "ygz/Basic.h"
#include "ygz/Algorithm.h"
#include <boost/timer.hpp>
// 这个程序测试BA.LobalBA的正确性
using namespace ygz;
using namespace std;

// 仿真 pose
SE3 keyframe_poses[8] = {
    SE3( SO3::exp(Vector3d(0,  0,  0  )), Vector3d(0,  0,  0  ) ), 
    SE3( SO3::exp(Vector3d(0.1,0,  0  )), Vector3d(0,  0,  0  ) ), 
    SE3( SO3::exp(Vector3d(0  ,0.1,0  )), Vector3d(0,  0,  0  ) ), 
    SE3( SO3::exp(Vector3d(0  ,0,  0.1)), Vector3d(0,  0,  0  ) ), 
    SE3( SO3::exp(Vector3d(0,  0,  0  )), Vector3d(0.1,0,  0  ) ), 
    SE3( SO3::exp(Vector3d(0,  0,  0  )), Vector3d(0,  0.1,0  ) ), 
    SE3( SO3::exp(Vector3d(0,  0,  0  )), Vector3d(0,  0,  0.1) ), 
    SE3( SO3::exp(Vector3d(0,  0,  0  )), Vector3d(0.1,0.1,0.1) ), 
};

Vector3d points[16] = {
    Vector3d(0,  0,  2), 
    Vector3d(0,  1,  2), 
    Vector3d(1,  0,  2), 
    Vector3d(1,  1,  2), 
    Vector3d(0,  0,  3), 
    Vector3d(0,  1,  3), 
    Vector3d(1,  0,  3), 
    Vector3d(1,  1,  3), 
    Vector3d(0,  0,  4), 
    Vector3d(0,  1,  4), 
    Vector3d(1,  0,  4), 
    Vector3d(1,  1,  4), 
    Vector3d(0,  0,  5), 
    Vector3d(0,  1,  5), 
    Vector3d(1,  0,  5), 
    Vector3d(1,  1,  5), 
};

int main( int argc, char** argv )
{
    set<Frame*> frames;
    set<MapPoint*> map_points;
    cv::RNG rng;
    
    ygz::Config::SetParameterFile("./config/default.yaml");
    ygz::PinholeCamera* cam = new ygz::PinholeCamera();
    ygz::Frame::SetCamera( cam );
    
    double noise_sigma = 1;
    
    vector<Frame*> frames_by_id;
    // create the key frames 
    for ( int i=0; i<8; i++ )
    {
        Frame* new_frame = new Frame();
        new_frame->_is_keyframe = true;
        Memory::RegisterKeyFrame( new_frame );
        Vector6d true_pose = keyframe_poses[i].log();
        Vector6d noisy_pose = true_pose;
        if ( i!=0 )
        {
            for ( size_t j=0; j<6; j++)
            {
                noisy_pose[j] = true_pose[j] + rng.gaussian(0.1);
            }
        }
        
        new_frame->_TCW = SE3::exp( noisy_pose );
        frames.insert(new_frame);
        
        frames_by_id.push_back( new_frame );
    }
    
    for ( int i=0; i<16; i++ )
    {
        MapPoint* mp = new MapPoint;
        mp->_id = i;
        mp->_pos_world = points[i];
        for ( size_t j=0; j<3; j++ )
        {
            mp->_pos_world[j] += rng.gaussian(0.1);
        }
        map_points.insert( mp );
        
        // set the observations
        for ( int j=0; j<8; j++ )
        {
            Feature* fea = new Feature(
                cam->World2Pixel( points[i], keyframe_poses[j] )
            );
            fea->_frame = frames_by_id[j];
            fea->_mappoint = mp;
            // add noise 
            fea->_pixel += Vector2d( rng.gaussian(noise_sigma), rng.gaussian(noise_sigma) );
            
            frames_by_id[j]->_features.push_back( fea );
            mp->_obs[j] = fea;
        }
        
        map_points.insert( mp );
    }
    
    // now lets optimize these frames and points 
    boost::timer timer;
    if ( argc == 2 && string(argv[1])==string("-g2o"))
        ba::LocalBAG2O( frames, map_points );
    else
        ba::LocalBA( frames, map_points );
    LOG(INFO)<<"local ba cost time: "<<timer.elapsed()<<endl;
    
    // compare the results 
    for ( Frame* f: frames ) 
    {
        LOG(INFO)<<"frame "<<f->_keyframe_id<<" estimated pose = \n"<<f->_TCW;
        LOG(INFO)<<"real pose = \n"<<keyframe_poses[f->_keyframe_id];
    }
    
    for ( MapPoint* mp: map_points )
    {
        LOG(INFO)<<"map point "<<mp->_id<<" estimated pos = "<<mp->_pos_world.transpose()<<endl;
        LOG(INFO)<<"real pos = "<<points[mp->_id].transpose()<<endl;
    }
    return 0;
}