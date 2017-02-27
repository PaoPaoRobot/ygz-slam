#include "ygz/Basic.h"
#include "ygz/Algorithm.h"

#include <boost/timer.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace std; 
using namespace cv; 

// 本程序测试地图点投影的结果

int main( int argc, char** argv )
{
    if ( argc< 2 ) {
        cout <<"usage: test_feature_alignment path_to_TUM_dataset [index1=0] " <<endl;
        return 1;
    }
    int index = 0;
    
    if ( argc == 3 )
        index = std::atoi(argv[2]);
    
    ifstream fin( string(argv[1])+"/associate.txt" ); 
    vector<string> rgbFiles, depthFiles;
    vector<double> rgbTime, depthTime; 
    while( !fin.eof() ) {
        double timeRGB, timeDepth;
        string rgbfile, depthfile;
        fin>>timeRGB>>rgbfile>>timeDepth>>depthfile;
        rgbFiles.push_back( rgbfile );
        depthFiles.push_back( depthfile );
        rgbTime.push_back(timeRGB);
        depthTime.push_back(timeDepth);
        if ( fin.good() == false ) 
            break;
    }
    fin.close();
    
    ygz::Config::SetParameterFile("./config/default.yaml");
    
    // read the first frame and create map points 
    LOG(INFO) << "Reading " << string(argv[1])+string("/")+rgbFiles[index];
    Mat color = imread( string(argv[1])+string("/")+rgbFiles[index] );
    Mat depth = imread( string(argv[1])+string("/")+depthFiles[index], -1 );
    ygz::FeatureDetector detector; 
    detector.LoadParams();
    
    // 这是reference 
    ygz::Frame* frame = new ygz::Frame(); 
    frame->_color = color;
    frame->_depth = depth;
    frame->InitFrame();
    ygz::Memory::RegisterKeyFrame( frame );
    detector.Detect( frame );
    
    ygz::PinholeCamera* cam = new ygz::PinholeCamera();
    ygz::Frame::SetCamera( cam );
    
    // init the map points using depth information 
    LOG(INFO)<<"creating map points"<<endl;
    int cnt_mp =0;
    for ( ygz::Feature* fea: frame->_features ) 
    {
        Vector2d pixel = fea->_pixel;
        unsigned short d = depth.ptr<ushort>( int(pixel[1]) )[int(pixel[0])];
        if ( d==0 ) 
            continue; 
        
        fea->_depth = double(d)/1000.0;
        ygz::MapPoint* mp = ygz::Memory::CreateMapPoint();
        mp->_pos_world = frame->_camera->Pixel2World( pixel, frame->_TCW, fea->_depth );
        fea->_mappoint = mp;
        mp->_obs[frame->_keyframe_id] = fea;
        cnt_mp++;
    }
    
    LOG(INFO) << "Set "<<cnt_mp<<" map points. "<<endl;
    
    // read the second frame 
    index +=2 ;
    LOG(INFO) << "Reading " << string(argv[1])+string("/")+rgbFiles[index];
    Mat color2 = imread( string(argv[1])+string("/")+rgbFiles[index] );
    ygz::Frame frame2;
    frame2._keyframe_id =1;
    frame2._color = color2;
    frame2.InitFrame();
    
    boost::timer timer;
    frame2._TCW = frame->_TCW;
    // Let's use SVO's sparse image alignment 
    // ygz::SparseImgAlign align( 2, 0, 
        // 30, ygz::SparseImgAlign::LevenbergMarquardt, true, true );
    
    ygz::SparseImgAlign align( 2, 0, 
        30, ygz::SparseImgAlign::GaussNewton, false, false );
    timer.restart();

    align.run( frame, &frame2 );
    LOG(INFO)<<"SVO's sparse image alignment costs time: "<<timer.elapsed()<<endl;
    
    ygz::Matcher matcher;
    // 求ref中地图点在frame2中的投影
    auto& all_points = ygz::Memory::GetAllPoints();
    vector<Vector2d> px_frame2, px_frame2_reproj;       // align之后和之前的（重投影的）像素位置
    timer.restart();
    int cnt_good_projection = 0;
    for ( auto& map_point: all_points )
    {
        ygz::MapPoint* mp = map_point.second;
        Vector2d px2 = frame2._camera->World2Pixel( mp->_pos_world, frame2._TCW );
        px_frame2_reproj.push_back(px2);
        int level=0;
        if ( matcher.FindDirectProjection( frame, &frame2, mp, px2,level ) )
        {
            cnt_good_projection++;
            
            ygz::Feature* new_feature = new ygz::Feature( px2 );
            new_feature->_level = level;
            frame2._features.push_back( new_feature );
            map_point.second->_obs[frame2._keyframe_id] = new_feature;
        }
        px_frame2.push_back(px2);
    }
    LOG(INFO) << "total points: "<<all_points.size()<<", succeed: "<<cnt_good_projection<<endl;
    LOG(INFO) << "project "<<all_points.size()<<" point cost time: "<<timer.elapsed()<<endl;
    
    // 比较描述子的距离
    detector.ComputeAngleAndDescriptor( &frame2 );
    for ( auto& mp: all_points )
    {
        if ( mp.second->_obs.size() == 2 )
        {
            // plot the matched features 
            Mat color1_show = frame->_color.clone();
            Mat color2_show = frame2._color.clone();
            
            ygz::Feature* f1 = mp.second->_obs[0];
            ygz::Feature* f2 = mp.second->_obs[1];
            
            double distance = matcher.DescriptorDistance( f1->_desc, f2->_desc );
            LOG(INFO)<<"distance = "<<distance<<endl;
            
            circle( color1_show, Point2f(f1->_pixel[0], f1->_pixel[1]), 
                2, Scalar(0,250,0), 2
            );
            
            circle( color2_show, Point2f(f2->_pixel[0], f2->_pixel[1]), 
                2, Scalar(0,250,0), 2
            );
            imshow("point in frame 1", color1_show );
            imshow("point in frame 2", color2_show );
            waitKey();
        }
    }
    
    destroyAllWindows();
    
    delete cam;
    ygz::Config::Release();
}
