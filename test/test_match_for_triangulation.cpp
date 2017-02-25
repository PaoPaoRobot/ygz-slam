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
    Mat depth2 = imread( string(argv[1])+string("/")+depthFiles[index2], -1 );
    ygz::Frame frame2;
    frame2._color = color2;
    frame2._depth = depth2;
    frame2.InitFrame();
    
    ygz::Matcher matcher;
    
    LOG(INFO)<<"doing sparse alignment"<<endl;
    boost::timer timer;
    matcher.SparseImageAlignment( frame, &frame2 );
    LOG(INFO)<<"Sparse image alignment costs time: "<<timer.elapsed()<<endl;
    SE3 TCR = matcher.GetTCR();
    LOG(INFO)<<"Estimated TCR: \n"<<TCR.matrix()<<endl;
    
    /*
    // 将Frame1的特征设置到Frame2中
    vector<pair<ygz::Feature*, ygz::Feature*>> copyed_features;
    for ( ygz::Feature* fea: frame->_features)
    {
        if ( fea->_mappoint )
        {
            Vector2d px2 = frame2._camera->World2Pixel( fea->_mappoint->_pos_world, TCR );
            ygz::Feature* feature2 = new ygz::Feature(
                px2,
                fea->_level,
                fea->_score
            );
            feature2->_frame = &frame2;
            feature2->_mappoint = fea->_mappoint;
            frame2._features.push_back( feature2 );
            
            copyed_features.push_back( make_pair(fea, feature2));
        }
    }
    detector.ComputeAngleAndDescriptor( &frame2 );
    */
    
    frame2._TCW = TCR;
    SE3 T12 = TCR.inverse();
    Eigen::Matrix3d E12 = SO3::hat( T12.translation() )*T12.rotation_matrix();
    
    // 再提一些新的
    detector.Detect( &frame2, false );
    
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
    
    
    // 其他颜色是匹配上的特征
    cv::RNG rng;
    for ( auto m: matched_points )
    {
        // plot the matched features 
        Mat color1_show = frame->_color.clone();
        Mat color2_show = frame2._color.clone();
        
        ygz::Feature* fea1 = frame->_features[m.first];
        ygz::Feature* fea2 = frame2._features[m.second];
        
        Scalar color( rng.uniform(0,255), rng.uniform(0,255), rng.uniform(0,255));
        circle( color1_show, 
            Point2f(frame->_features[m.first]->_pixel[0], frame->_features[m.first]->_pixel[1]), 
            1, Scalar(0,250,0), 2
        );
        
        circle( color2_show, 
            Point2f(frame2._features[m.second]->_pixel[0], frame2._features[m.second]->_pixel[1]), 
            1, Scalar(0,250,0), 2
        );
        
        Vector3d pt1 = frame->_camera->Pixel2Camera( frame->_features[m.first]->_pixel );
        Vector3d pt2 = frame2._camera->Pixel2Camera( frame2._features[m.second]->_pixel );
        double depth1, depth2;
        // 从depth from triangulation 获得粗略的深度估计
        bool ret = ygz::cvutils::DepthFromTriangulation( T12.inverse(), pt1, pt2, depth1, depth2, 1e-5 );
        
        
        ushort d1 = frame->_depth.ptr<ushort>( 
            cvRound(frame->_features[m.first]->_pixel[1]) )[cvRound(frame->_features[m.first]->_pixel[0])];
        ushort d2 = frame2._depth.ptr<ushort>( 
            cvRound(frame2._features[m.second]->_pixel[1]) )[cvRound(frame2._features[m.second]->_pixel[0])];
        
        if ( ret )
        {
            LOG(INFO)<<"Estimated depth = "<<depth1<<", real depth = "<<double(d1)/1000.0f << endl;;
            LOG(INFO)<<"Estimated depth = "<<depth2<<", real depth = "<<double(d2)/1000.0f << endl;;
            
            Vector2d px_curr = fea2->_pixel;
            fea1->_depth = depth1;
            fea2->_depth = depth2;
            int level = 0;
            matcher.FindDirectProjection( frame, &frame2, fea1, px_curr, level );
            
            // 蓝色点是修正后的点
            circle( color2_show, 
                Point2f(px_curr[0], px_curr[1]), 
                1, Scalar(250,0,0), 2
            );
            
            pt2 = frame2._camera->Pixel2Camera( px_curr );
            ret = ygz::cvutils::DepthFromTriangulation( T12.inverse(), pt1, pt2, depth1, depth2, 1e-5 );
            
            d2 = frame2._depth.ptr<ushort>
                ( cvRound( px_curr[1]) )[cvRound(px_curr[0])];
            LOG(INFO)<<"adjusted depth = "<<depth1<<", real depth = "<<double(d1)/1000.0f << endl;;
            LOG(INFO)<<"adjusted depth = "<<depth2<<", real depth = "<<double(d2)/1000.0f << endl;;
        }
        
        imshow("point in frame 1", color1_show );
        imshow("point in frame 2", color2_show );
        waitKey();
    }
    destroyAllWindows();
    
    delete cam;
}
