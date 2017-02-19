#include "ygz/Basic.h"
#include "ygz/Algorithm.h"

#include <boost/timer.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace std; 
using namespace cv; 

// 本程序测试 matcher::searchfortriangulation 结果
// 随意选取两张图，先用sparse image alignement 计算R,t，再用search for triangulation寻找可三角化的点

int main( int argc, char** argv )
{
    if ( argc< 2 ) {
        cout <<"usage: test_match_for_triangulation path_to_TUM_dataset [index1=0] [index2=index+1]" <<endl;
        return 1;
    }
    
    int index = 0;
    if ( argc == 3 )
        index = std::atoi( argv[2] );
    int index2 = index+1;
    if ( argc == 4 )
    {
        index = std::atoi( argv[2] );
        index2 = std::atoi( argv[3] );
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
    fin.close();
    
    ygz::Config::SetParameterFile("./config/default.yaml");
    
    // read the first frame and create map points 
    LOG(INFO) << "Reading " << string(argv[1])+string("/")+rgbFiles[index];
    Mat color = imread( string(argv[1])+string("/")+rgbFiles[index] );
    Mat depth = imread( string(argv[1])+string("/")+depthFiles[index], -1 );
    ygz::FeatureDetector detector; 
    detector.LoadParams();
    
    ygz::Frame* frame = new ygz::Frame(); 
    frame->_color = color;
    frame->_depth = depth;
    frame->InitFrame();
    ygz::Memory::RegisterKeyFrame( frame );
    detector.Detect( frame );
    
    ygz::PinholeCamera* cam = new ygz::PinholeCamera();
    ygz::Frame::SetCamera( cam );
    
    // init the map points using depth information 
    LOG(INFO)<<"creating map points"<<endl;
    int cnt_mp =0;
    for ( ygz::Feature* fea: frame->_features ) 
    {
        Vector2d pixel = fea->_pixel;
        unsigned short d = depth.ptr<ushort>( int(pixel[1]) )[int(pixel[0])];
        if ( d==0 ) 
            continue; 
        
        fea->_depth = double(d)/1000.0;
        ygz::MapPoint* mp = ygz::Memory::CreateMapPoint();
        mp->_pos_world = frame->_camera->Pixel2World( pixel, frame->_TCW, fea->_depth );
        fea->_mappoint = mp;
        mp->_obs[frame->_keyframe_id] = fea;
        cnt_mp++;
    }
    
    LOG(INFO) << "Set "<<cnt_mp<<" map points. "<<endl;
    
    // read the second frame 
    LOG(INFO) << "Reading " << string(argv[1])+string("/")+rgbFiles[index2];
    Mat color2 = imread( string(argv[1])+string("/")+rgbFiles[index2] );
    ygz::Frame frame2;
    frame2._color = color2;
    frame2.InitFrame();
    
    ygz::Matcher matcher;
    
    LOG(INFO)<<"doing sparse alignment"<<endl;
    boost::timer timer;
    matcher.SparseImageAlignment( frame, &frame2 );
    LOG(INFO)<<"Sparse image alignment costs time: "<<timer.elapsed()<<endl;
    SE3 TCR = matcher.GetTCR();
    LOG(INFO)<<"Estimated TCR: \n"<<TCR.matrix()<<endl;
    
    frame2._TCW = TCR;
    SE3 T12 = TCR.inverse();
    Eigen::Matrix3d E12 = SO3::hat( T12.translation() )*T12.rotation_matrix();
    
    detector.Detect( &frame2 );
    vector<pair<int, int>> matched_points;
    
    ORBVocabulary vocab;
    vocab.loadFromBinaryFile("./vocab/ORBvoc.bin");
    ygz::Frame::SetORBVocabulary(&vocab);
    frame->ComputeBoW();
    frame2.ComputeBoW();
    matcher.SearchForTriangulation( frame, &frame2, E12, matched_points );
    LOG(INFO)<<"search for triangulation matches: "<<matched_points.size()<<endl;
    
    map<int,int> matches_bow;
    matcher.SearchByBoW(frame, &frame2, matches_bow );
    LOG(INFO)<<"search by bow matches: "<<matches_bow.size()<<endl;
    
    // plot the matched features 
    Mat color1_show = frame->_color.clone();
    Mat color2_show = frame2._color.clone();
    
    for ( auto m: matched_points )
    {
        circle( color1_show, 
            Point2f(frame->_features[m.first]->_pixel[0], frame->_features[m.first]->_pixel[1]), 
            2, Scalar(0,250,0), 2
        );
        
        circle( color2_show, 
            Point2f(frame2._features[m.second]->_pixel[0], frame2._features[m.second]->_pixel[1]), 
            2, Scalar(0,250,0), 2
        );
    }
    imshow("point in frame 1", color1_show );
    imshow("point in frame 2", color2_show );
    waitKey();
    destroyAllWindows();
    
    delete cam;
}
