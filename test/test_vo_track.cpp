#include <fstream>
#include <string>
#include <iostream>

#include "ygz/Basic.h"
#include "ygz/Algorithm.h"
#include "ygz/Module/VisualOdometry.h"
#include "ygz/Module/LocalMapping.h"

using namespace std;
using namespace ygz;
using namespace cv;


/*****************************************
 * 测试tum上的VO程序的跟踪情况
 * 注意这个程序有内存泄漏，但暂时没有处理
 *****************************************/


int TestVOTrack::Main ( int argc, char** argv )
{
    if ( argc!= 2 ) {
        cout <<"usage: test_vo_init path_to_TUM_dataset " <<endl;
        return 1;
    }

    ifstream fin ( string ( argv[1] ) +"/associate.txt" );
    vector<string> rgbFiles, depthFiles;
    vector<double> rgbTime, depthTime;
    while ( !fin.eof() ) {
        double timeRGB, timeDepth;
        string rgbfile, depthfile;
        fin>>timeRGB>>rgbfile>>timeDepth>>depthfile;
        rgbFiles.push_back ( rgbfile );
        depthFiles.push_back ( depthfile );
        rgbTime.push_back ( timeRGB );
        depthTime.push_back ( timeDepth );
        if ( fin.good() == false ) {
            break;
        }
    }

    Config::SetParameterFile ( "./config/default.yaml" );
    PinholeCamera* cam = new PinholeCamera;
    Frame::SetCamera ( cam );
    VisualOdometry vo;

    ORBVocabulary vocab;
    vocab.loadFromBinaryFile ( "./vocab/ORBvoc.bin" );
    ygz::Frame::SetORBVocabulary ( &vocab );

    assert ( vo.GetStatus() == VisualOdometry::VO_NOT_READY );

    // set a key frame
    Mat color = imread ( string ( argv[1] ) +string ( "/" ) +rgbFiles[0] );
    Mat depth = imread ( string ( argv[1] ) +string ( "/" ) +depthFiles[0], -1 );

    ygz::Frame* frame = new ygz::Frame();
    frame->_color = color;
    frame->_depth = depth;
    frame->InitFrame();
    ygz::Memory::RegisterKeyFrame ( frame );
    frame->_is_keyframe = true;
    ygz::FeatureDetector detector;
    detector.LoadParams();
    detector.Detect ( frame );

    LOG ( INFO ) <<"creating map points"<<endl;
    int cnt_mp =0;
    
    for ( size_t i=0; i<frame->_features.size(); i++ ) {
        Feature*& fea= frame->_features[i];
        Vector2d pixel = fea->_pixel;
        unsigned short d = depth.ptr<ushort> ( int ( pixel[1] ) ) [int ( pixel[0] )];
        if ( d==0 || d>10000 ) {
            continue;
        }

        fea->_depth = double ( d ) /1000.0;
        ygz::MapPoint* mp = ygz::Memory::CreateMapPoint();
        mp->_pos_world = frame->_camera->Pixel2World ( pixel, frame->_TCW, fea->_depth );
        fea->_mappoint = mp;
        mp->_obs[frame->_keyframe_id] = fea;
        cnt_mp++;
    }
    frame->ComputeBoW();
    LOG ( INFO ) << "Set "<<cnt_mp<<" map points. "<<endl;

    // 手动设置第一个关键帧
    vo._status = VisualOdometry::VO_GOOD;
    vo._ref_frame = frame;
    vo._curr_frame = frame;
    vo._last_key_frame = frame;
    vo._local_mapping->UpdateLocalKeyframes ( frame );
    vo._local_mapping->UpdateLocalMapPoints ( frame );

    // track the following frames
    for ( size_t i=1; i<rgbFiles.size(); i++ ) {
        LOG ( INFO ) << "\nimage "<<i<<endl;
        Mat color = imread ( string ( argv[1] ) +string ( "/" ) +rgbFiles[i] );
        Mat depth = imread ( string ( argv[1] ) +string ( "/" ) +depthFiles[i],-1 );
        if ( color.data == nullptr ) {
            continue;
        }
        Frame* pf = new Frame;
        pf->_color = color.clone();
        pf->_depth = depth.clone();
        pf->InitFrame();

        /*
        if ( i == 29 )
        {
            // save the results and see what's wrong 
            LOG(INFO)<<"testing the 28th image"<<endl;
            Frame* ref = vo._ref_frame;
            
            LOG(INFO)<<"using vo's matcher"<<endl;
            vo._matcher->SparseImageAlignment( ref, pf );
            
            LOG(INFO)<<"using a new matcher"<<endl;
            Matcher matcher;
            matcher.SparseImageAlignment( ref, pf );
            
            LOG(INFO)<<"reset the feature's depth"<<endl;
            for ( Feature* fea: ref->_features )
            {
                if ( fea->_bad==false && fea->_mappoint && fea->_mappoint->_bad==false)
                {
                    if ( !ref->_depth.empty() )
                    {
                        unsigned short d = ref->_depth.ptr<ushort> ( int ( fea->_pixel[1] ) ) [int ( fea->_pixel[0] )];
                        if ( d==0 || d>10000 ) {
                            fea->_bad=true;
                            continue;
                        }
                        LOG(INFO)<<"estimated depth="<<fea->_depth<<", real depth = "<<d/1000.0<<endl;
                        fea->_depth = double ( d ) /1000.0;
                    }
                }
            }
            matcher.SparseImageAlignment( ref, pf );
            
            LOG(INFO)<<"using other features)"<<endl;
            vo._detector->Detect( ref );
            for ( Feature*& fea: ref->_features ) {
                Vector2d pixel = fea->_pixel;
                unsigned short d = ref->_depth.ptr<ushort> ( int ( pixel[1] ) ) [int ( pixel[0] )];
                if ( d==0 || d>10000 ) {
                    continue;
                }

                fea->_depth = double ( d ) /1000.0;
                ygz::MapPoint* mp = ygz::Memory::CreateMapPoint();
                mp->_pos_world = frame->_camera->Pixel2World ( pixel, frame->_TCW, fea->_depth );
                fea->_mappoint = mp;
                mp->_obs[frame->_keyframe_id] = fea;
            }
            matcher.SparseImageAlignment( ref, pf );
            
            Mat img_show = pf->_color.clone();
            Mat img_ref = ref->_color.clone();
            SE3 TCR = matcher.GetTCR();
            for ( Feature* fea : ref->_features )
            {
                if ( fea->_mappoint && fea->_bad==false )
                {
                    // Vector2d px = _curr_frame->_camera->World2Pixel( fea->_mappoint->_pos_world, _curr_frame->_TCW );
                    Vector2d px = pf->_camera->World2Pixel( 
                        ref->_camera->Pixel2Camera( fea->_pixel, fea->_depth ), TCR );
                    cv::circle( img_show, cv::Point2f(fea->_pixel[0],fea->_pixel[1]), 1, cv::Scalar(0,0,250), 2);
                    cv::circle( img_show, cv::Point2f(px[0],px[1]), 1, cv::Scalar(0,250,0), 2);
                    cv::circle( img_ref, cv::Point2f(fea->_pixel[0],fea->_pixel[1]), 1, cv::Scalar(0,0,250), 2);
                }
            }
            cv::imshow("Track ref frame: curr 28", img_show );
            cv::imshow("Track ref frame: ref 28", img_ref );
            cv::waitKey();
            break;
        }
        */
        
        bool ret = vo.AddFrame ( pf );
        if ( vo.GetStatus() == VisualOdometry::VO_GOOD ) {
            LOG ( INFO ) << "successfully tracked" <<endl;
            LOG (INFO) << "current pose = \n"<<pf->_TCW.matrix()<<endl;
            
            // Plot the track features
            Mat img_show = pf->_color.clone();
            // the map point projection 
            auto allpts = Memory::GetAllPoints();
            for ( auto pt: allpts )
            {
                Vector2d px = pf->_camera->World2Pixel( pt.second->_pos_world, pf->_TCW );
                circle ( img_show,
                    Point2f ( px[0], px[1] ),
                    1, Scalar ( 0,0,250 ),
                    2
                );
            }
            
            // and the tracked features 
            for ( Feature* fea: pf->_features ) {
                if ( fea->_mappoint )
                {
                    if ( fea->_bad == false )
                    {
                        // 正确投影的为绿色
                        circle ( img_show, Point2f ( fea->_pixel[0], fea->_pixel[1] ),
                                1, Scalar ( 0,250,0 ), 2 );
                    }
                    else
                    {
                        // 被BA剔除的点为蓝色
                        circle ( img_show, Point2f ( fea->_pixel[0], fea->_pixel[1] ),
                                1, Scalar ( 250,0,0 ), 2 );
                    }
                            
                }
            }

            imshow ( "tracked features", img_show );
            waitKey ( 0 );
        } else if ( vo._status == VisualOdometry::VO_LOST ) {
            break;
        }
        
    }

    delete cam;
    return 0;
}

int main ( int argc, char** argv )
{
    TestVOTrack t;
    return t.Main ( argc, argv );
}
