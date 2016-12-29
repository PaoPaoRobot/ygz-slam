#include <iostream>
#include <fstream>

#include "ygz/config.h"
#include "ygz/frame.h"
#include "ygz/ORB/ORBExtractor.h"
#include "ygz/feature_detector.h"

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std; 
using namespace cv; 

int main ( int argc, char** argv )
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
    
    ygz::Config::setParameterFile("./config/default.yaml");
    
    int index1 = 3;
    int index2 = 4;
    
    Mat color1 = imread( string(argv[1])+string("/")+rgbFiles[index1] );
    Mat color2 = imread( string(argv[1])+string("/")+rgbFiles[index2] );
    
    ygz::ORBExtractor orb;
    ygz::Frame frame1, frame2;
    frame1._color = color1;
    frame2._color = color2;
    frame1.InitFrame();
    frame2.InitFrame();
    
    ygz::FeatureDetector detector;
    detector.Detect( &frame1 );
    detector.Detect( &frame2 );
    
    orb.Compute( &frame1 );
    orb.Compute( &frame2 );
    
    // call bf matcher to match them 
    Mat desp1, desp2; 
    desp1 = frame1.GetAllDescriptors();
    desp2 = frame2.GetAllDescriptors();
    
    cout<<desp1<<endl;
    cout<<desp2<<endl;
    
    cv::BFMatcher matcher( cv::NORM_HAMMING, true );
    vector<DMatch> matches;
    matcher.match( desp1, desp2, matches );
    cout<<"matches: "<<matches.size()<<endl;
    
    for( DMatch& m:matches ) {
        cout<<"matches: "<<m.queryIdx<<","<<m.trainIdx<<endl;
    }
    
    vector<KeyPoint> kp1, kp2; 
    for ( KeyPoint& kp: frame1._map_point_candidates ) {
        kp1.push_back( kp );
    }
    for ( KeyPoint& kp: frame2._map_point_candidates ) {
        kp2.push_back( kp );
    }
    
    Mat img_show;
    cv::drawMatches( frame1._color, kp1, frame2._color, kp2, matches, img_show );
    imshow("matches", img_show);
    waitKey(0);
    
    // select good matches 
    float min_dis = min_element( matches.begin(), matches.end(), 
        [](const DMatch& m1, const DMatch& m2) {return m1.distance<m2.distance;}
    )->distance;
    cout<<"min dis = "<<min_dis<<endl;
    
    vector<DMatch> good;
    for ( DMatch& m:matches ) {
        if ( m.distance < 5.0*min_dis )
            good.push_back(m);
    }
    
    Mat img_show_good;
    cv::drawMatches( frame1._color, kp1, frame2._color, kp2, good, img_show_good );
    imshow("good matches", img_show_good);
    waitKey(0);
    
    
    // 原生ORB是什么样的？
    // origin ORB 
    kp1.clear();
    kp2.clear();
    Ptr<ORB> orb_cv = ORB::create(500, 2.0f, 3, 20 );
    desp1 = Mat(); desp2 = Mat(); 
    orb_cv->detect( frame1._color, kp1 );
    orb_cv->detect( frame2._color, kp2 );
    orb_cv->compute( frame1._color, kp1, desp1 );
    orb_cv->compute( frame2._color, kp2, desp2 );
    matcher.match( desp1, desp2, matches );
    Mat img_show_cv;
    cv::drawMatches( frame1._color, kp1, frame2._color, kp2, matches, img_show_cv );
    imshow("matches cv", img_show_cv );
    waitKey(0);
    
    
    
}