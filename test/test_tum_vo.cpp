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
    // google::InitGoogleLogging( argv[0] );
    google::InstallFailureSignalHandler();
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
    LOG(INFO) << "start viewer"<<endl;
    Viewer* v = new Viewer();
    
    for ( size_t i=0; i<rgbFiles.size(); i++ ) {
        Mat color = imread( string(argv[1])+string("/")+rgbFiles[i] );
        Frame::Ptr pf( new Frame );
        pf->_color = color.clone(); 
        pf->InitFrame();
        
        bool ret = vo.AddFrame( pf );
        
        if ( ret ) {
            LOG(INFO) << pf->_id<<endl;
            v->SetCurrPose( pf->_T_c_w );
            v->SetCurrFrame( pf );
            LOG(INFO) << v->_current->_id<<endl;
            LOG(INFO) << "viewer = " << v << endl;
            v->Draw();
            if ( v->_current )
                LOG(INFO) << v->_current->_id<<endl;
            cv::waitKey(0);
        }
    }
    delete v;
    
    return 0;
}