#include "ygz/tracker.h"
#include <fstream>

using namespace std;
using namespace ygz;
using namespace cv;

int main( int argc, char** argv ) 
{
    if ( argc!= 2 ) {
        LOG(IFNO) <<"usage: test_tum_tracking path_to_TUM_dataset " <<endl;
        return 1;
    }
    
    ifstream fin( argv[1]+"/associate.txt" ); 
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
    PinholeCamera::Ptr cam( new PinholeCamera ); 
    Frame::SetCamera( cam );
    
    Tracker tracker; 
    for ( size_t i=0; i<50; i++ ) {
        Mat color = imread( argv[1]+"/"+rgbFiles[i] );
        if ( i==0 ) {
            Frame::Ptr pf( new Frame );
            
            pf->SetCamera();
            
            tracker.SetReference( color );
            continue; 
        }
        
        tracker.Track();
        
    }
    
    return 0;
}