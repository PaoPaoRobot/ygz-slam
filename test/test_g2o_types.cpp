#include <fstream>
#include <string>
#include <iostream>

#include "ygz/Basic.h"
#include "ygz/Algorithm.h"
#include "ygz/Module/VisualOdometry.h"
#include "ygz/G2oTypes.h"
#include "ygz/Basic/Common.h"


using namespace std;
using namespace ygz;
using namespace cv;

/** modified
 * @px_valid_frame_2d frame1 valid 2D point pixels
 * @px_valid_frame2_2d frame2 valid 2D corresponding point pixels
 * @px_valid_3d_map 3d map corresponding points
 * @cam cam parameter 
 * @vec6d_result results
 **/
void TestEdgeSophusSE3ProjectXYZ(const vector<Vector2d> &px_valid_frame_2d,
    const vector<Vector2d> &px_valid_frame2_2d,
    const vector<MapPoint*> &px_valid_3d_map,
    const PinholeCamera* cam,
    Vector6d& vec6d_result);

/** 原始g2o边和顶点
 * @px_valid_frame_2d frame1 valid 2D point pixels
 * @px_valid_frame2_2d frame2 valid 2D corresponding point pixels
 * @px_valid_3d_map 3d map corresponding points
 * @cam cam parameter 
 * @vec6d_result results
 **/
void TestOrigin(const vector<Vector2d> &px_valid_frame_2d,
    const vector<Vector2d> &px_valid_frame2_2d,
    const vector<MapPoint*> &px_valid_3d_map,
    const PinholeCamera* cam,
    Vector6d& vec6d_result);

/** pose only
 * @px_valid_frame2_2d frame2 valid 2D corresponding point pixels
 * @px_valid_3d_map 3d map corresponding point pixels
 * @cam cam parameter 
 * @vec6d_result results
 **/
void TestEdgeSophusSE3ProjectXYZOnlyPose(const vector<Vector2d> &px_valid_frame2_2d,
    const vector<MapPoint*> &px_valid_3d_map,
    const PinholeCamera* cam,
    Vector6d& vec6d_result);


int main( int argc, char** argv )
{
    if ( argc != 4 ) {
        cout <<"usage: test_feature_alignment path_to_TUM_dataset [index1=0] " <<endl;
        return 1;
    }
    int index = 0;
    int index2 =0;
    
    if ( argc == 4 ){
      index = std::atoi(argv[2]);
      index2 = std::atoi(argv[3]);
    }
        
    
    // read the associate file
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
    
    // 这是reference 
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
    vector<Vector2d> px_frame;
    for ( ygz::Feature* fea: frame->_features ) 
    {
        Vector2d pixel = fea->_pixel;
        unsigned short d = depth.ptr<ushort>( int(pixel[1]) )[int(pixel[0])];
        if ( d==0 ) 
            continue; 
        px_frame.push_back(pixel);
        fea->_depth = double(d)/1000.0;
        ygz::MapPoint* mp = ygz::Memory::CreateMapPoint();
        mp->_pos_world = frame->_camera->Pixel2World( pixel, frame->_TCW, fea->_depth );
        fea->_mappoint = mp;
        mp->_obs[frame->_keyframe_id] = fea;
        cnt_mp++;
    }   
    LOG(INFO)<<"trans frame1: "<<frame->_TCW<<endl;  
    LOG(INFO) << "Set "<<cnt_mp<<" map points. "<<endl;
    
    // read the second frame 
    LOG(INFO) << "Reading " << string(argv[1])+string("/")+rgbFiles[index2];
    Mat color2 = imread( string(argv[1])+string("/")+rgbFiles[index2] );
    ygz::Frame frame2;
    frame2._color = color2;
    frame2.InitFrame();
    
    frame2._TCW = frame->_TCW;
    // Let's use SVO's sparse image alignment 
    // ygz::SparseImgAlign align( 2, 0, 
        // 30, ygz::SparseImgAlign::LevenbergMarquardt, true, true );
    
    ygz::SparseImgAlign align( 2, 0, 
        30, ygz::SparseImgAlign::GaussNewton, false, false );

    align.run( frame, &frame2 );
    LOG(INFO)<<"SVO results: "<<frame2._TCW.log()<<endl;
    
    ygz::Matcher matcher;
    // 求ref中地图点在frame2中的投影
    auto& all_points = ygz::Memory::GetAllPoints();

    // get the valid 3d points(map points) and the corresponding 2d points in frame and frame2
    vector<Vector2d> px_valid_frame_2d;
    vector<Vector2d> px_valid_frame2_2d;
    vector<MapPoint*> px_valid_3d_map;
    int cnt =0;
    for ( auto& map_point: all_points ){
        ygz::MapPoint* mp = map_point.second;
        Vector2d px2 = frame2._camera->World2Pixel( mp->_pos_world, frame2._TCW);
	int level = 0;
        if ( matcher.FindDirectProjection( frame, &frame2, mp, px2, level ) ){
	  px_valid_frame_2d.push_back(px_frame[cnt]);
	  px_valid_frame2_2d.push_back(px2);
	  px_valid_3d_map.push_back(mp);
	}
	cnt++;
    }
    LOG(INFO) << "total points: "<<all_points.size()<<", succeed: "<<px_valid_frame_2d.size()<<endl;
    
   
    // TestEdgeSophusSE3ProjectXYZ
    Vector6d edge_result;
    TestEdgeSophusSE3ProjectXYZ(px_valid_frame_2d,
				px_valid_frame2_2d,
				px_valid_3d_map,
				cam,
				edge_result);
    Mat color_show = frame2._color.clone();
    cnt = 0;
    for(auto& map_point: px_valid_3d_map)
    {
      ygz::MapPoint* mp = map_point;
      Vector6d result;
      result << edge_result[3],edge_result[4],edge_result[5],edge_result[0],edge_result[1],edge_result[2];
      Vector3d pos = SE3::exp(result)*(mp->_pos_world);
      Vector2d pose_pixel = cam->Camera2Pixel(pos);  
      Vector2d pose_ground_truth = px_valid_frame2_2d[cnt];
      
      circle( color_show, Point2f(pose_pixel[0], pose_pixel[1]), 1, Scalar(0,0,250), 2);
      circle( color_show, Point2f(pose_ground_truth[0], pose_ground_truth[1]), 1, Scalar(0,250,0), 2); 
      cnt++;
    }
    imshow("TestEdgeSophusSE3ProjectXYZ(frame 2)", color_show );
    waitKey();
    destroyAllWindows();
    
    
    // TestEdgeSophusSE3ProjectXYZOnlyPose
    Vector6d poseOnly_result;
    TestEdgeSophusSE3ProjectXYZOnlyPose(px_valid_frame2_2d,
				  px_valid_3d_map,
				  cam,
				  poseOnly_result);
    Mat color2_show = frame2._color.clone();
    cnt = 0;
    for(auto& map_point: px_valid_3d_map)
    {
      ygz::MapPoint* mp = map_point;
      Vector2d px2 = px_valid_frame2_2d[cnt]; 
      Vector6d result;
      result << poseOnly_result[3],poseOnly_result[4],poseOnly_result[5],poseOnly_result[0],poseOnly_result[1],poseOnly_result[2];
      Vector3d pos = SE3::exp(result)*(mp->_pos_world);
      Vector2d pose_pixel = cam->Camera2Pixel(pos);
      Vector2d pose_ground_truth = px_valid_frame2_2d[cnt];
    
      circle( color2_show, Point2f(pose_pixel[0], pose_pixel[1]), 1, Scalar(0,0,250), 2);
      circle( color2_show, Point2f(pose_ground_truth[0], pose_ground_truth[1]), 1, Scalar(0,250,0), 2); 
      cnt++;
    }
    imshow("TestEdgeSophusSE3ProjectXYZOnlyPose(frame 2)", color2_show );
    waitKey();
    destroyAllWindows();
    
    
    delete cam;
    ygz::Config::Release();
}

void TestEdgeSophusSE3ProjectXYZ(const vector<Vector2d> &px_valid_frame_2d,
    const vector<Vector2d> &px_valid_frame2_2d,
    const vector<MapPoint*> &px_valid_3d_map,
    const PinholeCamera* cam,
    Vector6d& vec6d_result){
  
    // test g2o
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();
    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose( false );
    
    // add vertices, frames
    for(size_t i =0; i< 2;i++)
    {
      ygz::VertexSE3Sophus* vSE3 = new ygz::VertexSE3Sophus(); 
      vSE3->setFixed(i==0);
      vSE3->setToOriginImpl();	
      vSE3->setId(i);
      optimizer.addVertex(vSE3);    
    }
       
    // add vertices, 3D map points
    for ( auto& map_point: px_valid_3d_map )
    {
      g2o::VertexSBAPointXYZ* v = new g2o::VertexSBAPointXYZ();
      ygz::MapPoint* mp = map_point;
      v->setId(mp->_id+2);
      v->setFixed(true);
      v->setEstimate(mp->_pos_world);
      v->setMarginalized(true);
      optimizer.addVertex(v);
    }
    
    // add edges: 3DPoint<->frame1
    int cnt = 0;
    for(auto& map_point: px_valid_3d_map)
    {
      ygz::MapPoint* mp = map_point;
      ygz::EdgeSophusSE3ProjectXYZ* edge = new ygz::EdgeSophusSE3ProjectXYZ();
      // camera
      edge->setCamera(cam);
      
      edge->setVertex(0,dynamic_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(mp->_id+2)));
      edge->setVertex(1,dynamic_cast<ygz::VertexSE3Sophus*>(optimizer.vertex(0)));
      
      edge->setMeasurement(px_valid_frame_2d[cnt++]);
      edge->setInformation(Eigen::Matrix2d::Identity());
      edge->setRobustKernel( new g2o::RobustKernelHuber() );
      // edge->setParameterId(0,0);
      optimizer.addEdge( edge );
    }
    LOG(INFO)<<"3DPoint<->frame1 num: "<<cnt<<endl;
    
    // add edges: 3DPoint<->frame2
    cnt = 0;
    for(auto& map_point: px_valid_3d_map)
    {
      ygz::MapPoint* mp = map_point;
      ygz::EdgeSophusSE3ProjectXYZ* edge = new ygz::EdgeSophusSE3ProjectXYZ();
      // camera
      edge->setCamera(cam);
      
      edge->setVertex(0,dynamic_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(mp->_id+2)));
      edge->setVertex(1,dynamic_cast<ygz::VertexSE3Sophus*>(optimizer.vertex(1)));
      edge->setMeasurement(px_valid_frame2_2d[cnt++]);
      edge->setInformation(Eigen::Matrix2d::Identity());
      edge->setParameterId(0,0);
      edge->setRobustKernel( new g2o::RobustKernelHuber() );
      optimizer.addEdge( edge );
    }
    LOG(INFO)<<"3DPoint<->frame2 num: "<<cnt<<endl;
    
    // Optimize!
    LOG(INFO)<<"Vertices num: "<<optimizer.vertices().size()<<endl;
    LOG(INFO)<<"   Edges num: "<<optimizer.edges().size()<<endl;
    optimizer.setVerbose( true );
    optimizer.initializeOptimization();
    optimizer.optimize(20);
    
    // get the result
    ygz::VertexSE3Sophus* v = dynamic_cast<ygz::VertexSE3Sophus*>( optimizer.vertex(1) );
    LOG(INFO)<<"Pose = "<<v->estimate()<<endl;
    vec6d_result = v->estimate();
    }


void TestEdgeSophusSE3ProjectXYZOnlyPose(const vector<Vector2d> &px_valid_frame2_2d,
    const vector<MapPoint*> &px_valid_3d_map,
    const PinholeCamera* cam,
    Vector6d& vec6d_result){
  // create solver
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;
    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();
    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose( false );
    
    // add pose to the graph
    ygz::VertexSE3Sophus* vSE3 = new ygz::VertexSE3Sophus(); 
    vSE3->setFixed(false);
    vSE3->setToOriginImpl();	
    vSE3->setId(0);
    optimizer.addVertex(vSE3);     
    
    // add poseonly edge
    int cnt = 0;
    for ( auto& map_point: px_valid_3d_map )
    {
      ygz::EdgeSophusSE3ProjectXYZOnlyPose* e = new ygz::EdgeSophusSE3ProjectXYZOnlyPose();
      
      e->setCamera(cam);
      e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
      Vector2d obs(px_valid_frame2_2d[cnt]);
      e->setMeasurement(obs);
      e->setInformation(Eigen::Matrix2d::Identity());
      e->setRobustKernel( new g2o::RobustKernelHuber() );
      e->setParameterId(0,0);     
      ygz::MapPoint* mp = map_point;
      e->Xw = mp->_pos_world;

      optimizer.addEdge(e);
      cnt++;
    }
     
    // start optimize
    optimizer.setVerbose( true );
    optimizer.initializeOptimization();
    optimizer.optimize(20);
    
    // get the result
    ygz::VertexSE3Sophus* v = dynamic_cast<ygz::VertexSE3Sophus*>( optimizer.vertex(0) );
    LOG(INFO)<<"Pose = "<<v->estimate()<<endl;
    vec6d_result = v->estimate();
}

void TestOrigin(const vector<Vector2d> &px_valid_frame_2d,
    const vector<Vector2d> &px_valid_frame2_2d,
    const vector<MapPoint*> &px_valid_3d_map,
    const PinholeCamera* cam,
    Vector6d& vec6d_result){
  // test g2o
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();
    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose( false );
    
    g2o::CameraParameters* camera = new g2o::CameraParameters( cam->fx(), Eigen::Vector2d(cam->cx(), cam->cy()), 0 );
    camera->setId(0);
    optimizer.addParameter( camera );
    
    // add vertices, frames
    for(size_t i =0; i< 2;i++)
    {
      g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap(); 
      vSE3->setFixed(i==0);
      vSE3->setToOriginImpl();	
      vSE3->setId(i);
      optimizer.addVertex(vSE3);    
    }
       
    // add vertices, 3D map points
    for ( auto& map_point: px_valid_3d_map )
    {
      g2o::VertexSBAPointXYZ* v = new g2o::VertexSBAPointXYZ();
      ygz::MapPoint* mp = map_point;
      v->setId(mp->_id+2);
      v->setFixed(true);
      v->setEstimate(mp->_pos_world);
      v->setMarginalized(true);
      optimizer.addVertex(v);
    }
    
    // add edges: 3DPoint<->frame1
    int cnt = 0;
    for(auto& map_point: px_valid_3d_map)
    {
      ygz::MapPoint* mp = map_point;
      g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
      // camera
      
      edge->setVertex(0,dynamic_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(mp->_id+2)));
      edge->setVertex(1,dynamic_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0)));
      
      edge->setMeasurement(px_valid_frame_2d[cnt++]);
      edge->setInformation(Eigen::Matrix2d::Identity());
      edge->setRobustKernel( new g2o::RobustKernelHuber() );
      edge->setParameterId(0,0);
      optimizer.addEdge( edge );
    }
    LOG(INFO)<<"3DPoint<->frame1 num: "<<cnt<<endl;
    
    // add edges: 3DPoint<->frame2
    cnt = 0;
    for(auto& map_point: px_valid_3d_map)
    {
      ygz::MapPoint* mp = map_point;
      g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
      
      edge->setVertex(0,dynamic_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(mp->_id+2)));
      edge->setVertex(1,dynamic_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(1)));
      edge->setMeasurement(px_valid_frame2_2d[cnt++]);
      edge->setInformation(Eigen::Matrix2d::Identity());
      edge->setParameterId(0,0);
      edge->setRobustKernel( new g2o::RobustKernelHuber() );
      optimizer.addEdge( edge );
    }
    LOG(INFO)<<"3DPoint<->frame2 num: "<<cnt<<endl;
    
    // Optimize!
    LOG(INFO)<<"Vertices num: "<<optimizer.vertices().size()<<endl;
    LOG(INFO)<<"   Edges num: "<<optimizer.edges().size()<<endl;
    optimizer.setVerbose( true );
    optimizer.initializeOptimization();
    optimizer.optimize(20);
    
    // get the result
    g2o::VertexSE3Expmap* v = dynamic_cast<g2o::VertexSE3Expmap*>( optimizer.vertex(1) );
    LOG(INFO)<<"Pose = "<<v->estimate()<<endl; 
    Vector6d vec6d_tmp= v->estimate().log();
    for(int i =0;i<3;i++){
     vec6d_result[i] = vec6d_tmp[i+3]; 
    }
    for(int i =0;i<3;i++){
     vec6d_result[i+3] = vec6d_tmp[i]; 
    }
}