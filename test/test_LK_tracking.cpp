#include <fstream>
#include <string>
#include <iostream>

#include "ygz/Basic.h"
#include "ygz/Algorithm.h"

using namespace std;
using namespace cv;

// 本程序测试LK光流的追踪效果
// 在两图像相距较大时，光流无法保证每个追踪点都是正确的

int main( int argc, char** argv ) 
{
    if ( argc< 2 ) {
        cout <<"usage: test_LK_tracking path_to_TUM_dataset [images=50]" <<endl;
        return 1;
    }
    
    int max_images = 50;
    if ( argc == 3 )
        max_images = std::atoi(argv[2]);
    
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
    
    ygz::Config::SetParameterFile("./config/default.yaml");
    ygz::PinholeCamera* cam = new ygz::PinholeCamera; 
    ygz::Frame::SetCamera( cam );
    
    LOG(INFO)<<"Set reference frame."<<endl;
    ygz::Tracker tracker; 
    
    // set the reference 
    Mat color = imread( string(argv[1])+string("/")+rgbFiles[0] );
    ygz::Frame* pf = new ygz::Frame;
    pf->_color = color;
    pf->InitFrame();
    
    ygz::FeatureDetector detector;
    detector.LoadParams();
    
    detector.Detect( pf );
    tracker.SetReference( pf );
    
    // track the following 50 frames
    for ( size_t i=1; i<max_images; i++ ) {
        Mat color = imread( string(argv[1])+string("/")+rgbFiles[i] );
        ygz::Frame* p = new ygz::Frame;
        p->_color = color; 
        p->InitFrame();
        tracker.Track( p );
        tracker.PlotTrackedPoints();
        if ( tracker.Status() == ygz::Tracker::TRACK_LOST )
            break; 
        LOG(INFO)<<"Mean disparity = "<<tracker.MeanDisparity()<<endl;
        
        delete p;
    }
    
    delete cam;
    return 0;
}