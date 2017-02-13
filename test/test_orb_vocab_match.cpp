#include <iostream>
#include <fstream>

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <boost/timer.hpp>

#include "ygz/config.h"
#include "ygz/frame.h"
#include "ygz/ORB/ORBExtractor.h"
#include "ygz/ORB/ORBMatcher.h"
#include "ygz/ORB/ORBVocabulary.h"
#include "ygz/feature_detector.h"

using namespace std; 
using namespace cv; 

/******************************
 * 测试 orb 字典是否正确
 * ****************************/

int main( int argc, char** argv ) 
{
    ygz::Config::setParameterFile("./config/default.yaml");
    if ( argc< 2 ) {
        cout <<"usage: test_orb_vocab_match path_to_TUM_dataset [index1] [index2]" <<endl;
        return 1;
    }
    int index1 = 1;
    int index2 = 2;
    
    if ( argc == 3 )
        index1 = std::atoi(argv[2]);
    if ( argc == 4 ) {
        index1 = std::atoi(argv[2]);
        index2 = std::atoi(argv[3]);
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
    
    Mat color1 = imread( string(argv[1])+string("/")+rgbFiles[index1] );
    Mat color2 = imread( string(argv[1])+string("/")+rgbFiles[index2] );
    
    ygz::ORBExtractor orb;
    ygz::Frame frame1, frame2;
    frame1._color = color1;
    frame2._color = color2;
    frame1.InitFrame();
    frame2.InitFrame();
    
    ygz::ORBVocabulary vocab; 
    cout<<"load from binary file"<<endl;
    vocab.loadFromBinaryFile("./vocab/ORBvoc.bin");
    cout<<vocab<<endl;
    ygz::Frame::_vocab = &vocab; 
    
    ygz::FeatureDetector detector;
    detector.Detect( &frame1 );
    detector.Detect( &frame2 );
    
    orb.Compute( &frame1 );
    orb.Compute( &frame2 );
    
    // the bow features 
    frame1.ComputeBoW();
    frame2.ComputeBoW();
    
    // match by bow 
    ygz::ORBMatcher matcher; 
    map<int,int> matches; 
    cout<<"matching descriptors ... "<<endl;
    boost::timer timer;
    matcher.SearchByBoW( &frame1, &frame2, matches );
    cout<<"match cost time: "<<timer.elapsed()<<endl;
    cout<<"matchces: "<<matches.size()<<endl;
    
    vector<DMatch> matches_cv;
    for ( auto& m: matches ) {
        DMatch mm;
        mm.queryIdx = m.first;
        mm.trainIdx = m.second;
        matches_cv.push_back(mm);
    }
    
    Mat img;
    drawMatches( frame1._color, frame1._map_point_candidates, frame2._color, frame2._map_point_candidates, matches_cv,img );
    imshow("matches", img );
    waitKey(0);
    
    return 0;
}