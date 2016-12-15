#include <fstream>
#include <string>
#include <iostream>

#include "ygz/visual_odometry.h"
#include "ygz/viewer.h"

using namespace std;
using namespace ygz;
using namespace cv;


/*****************************************
 * 测试tum上的VO程序 
 *****************************************/

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
    PinholeCamera::Ptr cam( new PinholeCamera ); 
    Frame::SetCamera( cam );
    VisualOdometry vo(nullptr); 
    Viewer viewer;
    
    for ( size_t i=0; i<rgbFiles.size(); i++ ) {
        Mat color = imread( string(argv[1])+string("/")+rgbFiles[i] );
        Frame::Ptr pf( new Frame );
        pf->_color = color.clone(); 
        pf->InitFrame();
        
        // cv::imshow("pyramid[0]", pf->_pyramid[0]);
        // cv::waitKey(1);
        
        bool ret = vo.AddFrame( pf );
        // vo.PlotFrame();
        
        if ( ret ) {
            // viewer.AddTempPose( pf->_T_c_w );
            viewer.SetCurrPose( pf->_T_c_w );
        }
        viewer.Draw();
    }
    
    return 0;
}