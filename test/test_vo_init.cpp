#include <fstream>
#include <string>
#include <iostream>

#include "ygz/Basic.h"
#include "ygz/Algorithm.h"
#include "ygz/Module/VisualOdometry.h"

using namespace std;
using namespace ygz;
using namespace cv;


/*****************************************
 * 测试tum上的VO程序的初始化情况
 * 注意这个程序有内存泄漏，但暂时没有处理
 *****************************************/

int main( int argc, char** argv ) 
{
    if ( argc!= 2 ) {
        cout <<"usage: test_vo_init path_to_TUM_dataset " <<endl;
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
    
    Config::SetParameterFile("./config/default.yaml");
    PinholeCamera* cam = new PinholeCamera; 
    Frame::SetCamera( cam );
    VisualOdometry vo; 
    
    ORBVocabulary vocab;
    vocab.loadFromBinaryFile("./vocab/ORBvoc.bin");
    ygz::Frame::SetORBVocabulary(&vocab);
    
    assert( vo.GetStatus() == VisualOdometry::VO_NOT_READY );
    
    // test the initialization 
    for ( size_t i=0; i<rgbFiles.size(); i++ ) {
        LOG(INFO) << "image "<<i<<endl;
        Mat color = imread( string(argv[1])+string("/")+rgbFiles[i] );
        if ( color.data == nullptr ) 
            continue;
        Frame* pf = new Frame;
        pf->_color = color.clone(); 
        pf->InitFrame();
        
        bool ret = vo.AddFrame( pf );
        if ( vo.GetStatus() == VisualOdometry::VO_GOOD )
        {
            LOG(INFO) << "VO is successfully initialized"<<endl;
            break;
        }
    }
    
    // plot the results 
    Frame* ref = vo.GetRefFrame();
    Frame* curr = vo.GetCurrFrame();
    assert( ref!=nullptr && curr!=nullptr );
    
    Mat img_ref = ref->_color.clone();
    Mat img_curr = curr->_color.clone();
    
    int cnt_ref = 0;
    for ( Feature* fea: ref->_features ) 
    {
        if ( fea->_mappoint == nullptr || fea->_mappoint->_bad==true )
        {
            circle( img_ref, 
                Point2f( fea->_pixel[0], fea->_pixel[1]),
                2, Scalar(0,0,0),
                2
            );
        }
        else 
        {
            // plot the distance 
            int c = 128+100*(fea->_depth-1);
            circle(
                img_ref, 
                Point2f( fea->_pixel[0], fea->_pixel[1]),
                2, 
                Scalar(c,c, 255),
                2
            );
            cnt_ref++;
        }
    }
    
    // and the current 
    int cnt_current =0;
    for ( Feature* fea: curr->_features ) 
    {
        if ( fea->_mappoint == nullptr || fea->_mappoint->_bad==true )
        {
            circle( img_curr, 
                Point2f( fea->_pixel[0], fea->_pixel[1]),
                2, Scalar(0,0,0),
                2
            );
        }
        else 
        {
            // plot the distance 
            int c = 128+100*(fea->_depth-1);
            circle(
                img_curr, 
                Point2f( fea->_pixel[0], fea->_pixel[1]),
                2, 
                Scalar(c,c,255),
                2
            );
            cnt_current ++;
        }
    }
    
    LOG(INFO)<<"triangulated features: ref = "<<cnt_ref<<", curr = "<<cnt_current<<endl;
    
    imshow("Reference", img_ref );
    imshow("Current", img_curr );
    waitKey(0);
    
    delete cam;
    
    return 0;
}