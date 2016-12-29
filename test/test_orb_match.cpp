#include <iostream>
#include <fstream>

#include "ygz/config.h"
#include "ygz/frame.h"
#include "ygz/ORB/ORBExtractor.h"
#include "ygz/feature_detector.h"

#include <opencv2/features2d/features2d.hpp>

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
    
    int index = 1;
    Mat color = imread( string(argv[1])+string("/")+rgbFiles[index] );
    
    // original orb 
    Ptr<ORB> orb_cv = ORB::create( 500, 2.0f, 3 );
    vector<KeyPoint> keypoints_cv;
    Mat descriptors_cv; 
    orb_cv->detectAndCompute( color, Mat(), keypoints_cv, descriptors_cv );
    // orb_cv->descriptorSize();
    // (*orb_cv)( color, Mat(), keypoints_cv );
    
    // changed orb
    ygz::Frame frame; 
    frame._color = color;
    frame.InitFrame();
    
    ygz::FeatureDetector detector;
    detector.Detect( &frame );
    ygz::ORBExtractor orb_ygz; 
    orb_ygz.Compute( &frame );
    
    Mat features_cv, features_ygz;
    cv::drawKeypoints( frame._color, keypoints_cv, features_cv );
    vector<KeyPoint> kp_ygz; 
    for ( KeyPoint& kp: frame._map_point_candidates )
        kp_ygz.push_back( kp );
    
    cv::drawKeypoints( frame._color, kp_ygz, features_ygz );
    
    // cout<<descriptors_cv<<endl;
    Mat desp_ygz( frame._map_point_candidates.size(), 32, frame._descriptors.front().type() );
    int i =0;
    for ( Mat& desp: frame._descriptors )  {
        desp.copyTo( desp_ygz.row(i++) );
        // cout<<desp<<endl;
    }
    
    cout<<desp_ygz.rows<<","<<desp_ygz.cols<<endl;
    
    imshow("features cv", features_cv );
    imshow("features ygz", features_ygz );
    waitKey(0);
    
    // compute the descriptors given the keypoints from ygz 
    Mat desp_from_ygz_keypoints;
    orb_cv->compute( frame._color, kp_ygz, desp_from_ygz_keypoints );
    
    cout<<desp_from_ygz_keypoints.row(0) << endl;
    cout<<frame._descriptors.front()<<endl;
    
    // match them ? 
    cv::BFMatcher matcher;
    vector<DMatch> matches;
    matcher.match( desp_ygz, desp_from_ygz_keypoints, matches );
    cout<<"matches: "<<matches.size()<<endl;
    
    for( DMatch& m:matches ) {
        cout<<"matches: "<<m.queryIdx<<","<<m.trainIdx<<endl;
    }
    /*
    Mat img_show;
    cv::drawMatches( frame._color, kp_ygz, frame._color, kp_ygz, matches, img_show );
    imshow("matches", img_show);
    waitKey(0);
    */
}