#include <iostream>
#include <fstream>


#include <boost/timer.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "ygz/Basic.h"
#include "ygz/Algorithm.h"

using namespace std; 
using namespace cv; 

int main( int argc, char** argv ) 
{
    if ( argc< 2 ) {
        cout <<"usage: test_feature_extraction path_to_TUM_dataset [index1=0] " <<endl;
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
    
    ygz::Config::SetParameterFile("./config/default.yaml");
    
    
    Mat color = imread( string(argv[1])+string("/")+rgbFiles[index] );
    
    ygz::FeatureDetector detector; 
    detector.LoadParams();
    
    ygz::Frame frame; 
    frame._color = color;
    frame.InitFrame();
    
    boost::timer timer; 
    detector.Detect( &frame );
    cout<<"detect cost time: "<<timer.elapsed()<<endl;
    cout<<"total features: "<<frame._features.size()<<endl;
    
    // and original ORB in OpenCV 
    
    Ptr<ORB> orb_cv = ORB::create(frame._features.size(), 2.0f, 3, 20 );
    Mat desp;
    vector<cv::KeyPoint> kps;
    
    timer.restart();
    orb_cv->detectAndCompute( color, Mat(), kps, desp);
    cout<<"OpenCV detect cost time: "<<timer.elapsed()<<endl;
    
    // plot the features 
    Mat show = color.clone();
    for ( ygz::Feature* fea: frame._features )
    {
        circle( show, Point2f(fea->_pixel[0],fea->_pixel[1]), 2, Scalar(0,250,0),2);
    }
    imshow("Features in ygz", show );
    
    Mat show_cv = color.clone();
    drawKeypoints( color, kps, show_cv );
    imshow("Features in cv", show_cv );
    waitKey();
    
    frame.CleanAllFeatures();
}