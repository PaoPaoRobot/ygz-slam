#include <fstream>
#include <string>
#include <iostream>

#include "ygz/tracker.h"
#include "ygz/config.h"
#include "ygz/camera.h"
#include "ygz/feature_detector.h"


using namespace std;
using namespace ygz;
using namespace cv;

int main( int argc, char** argv ) 
{
    if ( argc!= 2 ) {
        cout <<"usage: test_tum_tracking path_to_TUM_dataset " <<endl;
        return 1;
    }
    
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
    
    Config::setParameterFile("./config/default.yaml");
    PinholeCamera* cam = new PinholeCamera; 
    Frame::SetCamera( cam );
    
    Tracker tracker; 
    for ( size_t i=0; i<50; i++ ) {
        Mat color = imread( string(argv[1])+string("/")+rgbFiles[i] );
        Frame* pf = new Frame ;
        pf->_color = color; 
        pf->InitFrame();
        if ( i==0 ) {
            tracker.SetReference( pf );
            continue; 
        }
        
        tracker.Track( pf );
        tracker.PlotTrackedPoints();
        if ( tracker.Status() == Tracker::TRACK_LOST )
            break; 
        
        // delete就交给visual odometry自己决定吧
    }
    
    delete cam;
    
    return 0;
}