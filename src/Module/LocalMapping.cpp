#include "ygz/Module/LocalMapping.h"

namespace ygz
{
    
LocalMapping::LocalMapping() 
{
    _matcher = new Matcher;
    
}
    
void LocalMapping::AddKeyFrame( Frame* keyframe )
{
    assert( keyframe->_is_keyframe == true );
    _new_keyframes.push_back( keyframe );
}

void LocalMapping::AddMapPoint ( MapPoint* mp )
{
    assert(mp !=nullptr);
    _local_map_points.insert( mp );
}

bool LocalMapping::TrackLocalMap( Frame* current )
{
    LOG(INFO)<<"Local keyframes: "<<_local_keyframes.size()<<", map points: "<<_local_map_points.size()<<endl;
    // 寻找候选点
    map<Feature*, Vector2d> candidates = FindCandidates( current );
    LOG(INFO)<<"find total "<<candidates.size()<<" candidates."<<endl;
    
    // 投影候选点
    ProjectMapPoints( current, candidates );
    LOG(INFO)<<"project total "<<current->_features.size()<<" points"<<endl;
    
    // optimize the current pose and structure, remove outliers 
    OptimizeCurrent( current );
    LOG(INFO)<<"optimized pose = \n"<<current->_TCW.matrix()<<endl;
    
    int cnt=0;
    for ( Feature* fea:current->_features )
        if ( fea->_bad==false ) 
            cnt++;
    
    return cnt>_options._min_track_local_map_inliers;
}

std::map<Feature*, Vector2d> LocalMapping::FindCandidates(Frame* current)
{
    map<Feature*, Vector2d> candidates; // 每个特征以及对应的投影位置
    int cnt_candidate =0;
    // 遍历local map points，寻找在当前帧视野范围内，而且能匹配上patch的那些点
    for ( auto it = _local_map_points.begin(); it!=_local_map_points.end(); it++ )
    {
        MapPoint* map_point = *it;
        if ( map_point->_bad == true )
            continue;
        
        Vector3d pt_curr = current->_camera->World2Camera( map_point->_pos_world, current->_TCW );
        Vector2d px_curr = current->_camera->Camera2Pixel( pt_curr );
        if ( pt_curr[2] < 0 || !current->InFrame(px_curr,20) ) { // 在相机后面或不在视野内 
            map_point->_track_in_view = false;
            continue;
        }
        
        map_point->_cnt_visible++;
        
        // 检查local frame中有没有共视的
        for ( auto& obs_pair: map_point->_obs ) 
        {
            if ( _local_keyframes.find(obs_pair.second->_frame) 
                != _local_keyframes.end() )
            {
                // 有共视，设为候选
                cnt_candidate++;
                candidates[obs_pair.second] = px_curr;
            }
        }
    }
    return candidates;
}
    
void LocalMapping::ProjectMapPoints(
    Frame* current, 
    std::map< Feature*, Vector2d >& candidates)
{
    set<MapPoint*> matched_mps; // matched map points 
    for ( auto& candidate: candidates )
    {
        if ( matched_mps.find(candidate.first->_mappoint) != matched_mps.end() )
            continue;   // already matched 
        int level=0;
        
        if ( candidate.first->_mappoint->_first_seen > 0 )
        {
            // LOG(INFO)<<"this is a newly created map point: "<<candidate.first->_mappoint->_id<<endl;
        }
        
        bool ret = _matcher->FindDirectProjection(
            candidate.first->_frame, 
            current, 
            candidate.first->_mappoint,
            candidate.second,
            level
        );
        
        if ( ret == true ) 
        {
            matched_mps.insert( candidate.first->_mappoint );
            // add a new feature in current 
            Feature* feature = new Feature(
                candidate.second, 
                level,
                candidate.first->_score
            );
            feature->_frame = current;
            feature->_mappoint = candidate.first->_mappoint;
            current->_features.push_back( feature );
            feature->_mappoint->_cnt_found++;
        }
    }
}

void LocalMapping::OptimizeCurrent(Frame* current)
{
    // use BA to optimize the current frame's pose
    // ba::OptimizeCurrent( current );
    ba::OptimizeCurrentPoseOnly( current );
    // ba::OptimizeCurrentPointOnly( current );
    
    // reset the depth of current features
    /*
    for ( Feature* fea: current->_features )
    {
        if ( fea->_bad==false && fea->_mappoint && fea->_mappoint->_bad==false)
        {
            fea->_depth = current->_camera->World2Camera( fea->_mappoint->_pos_world, current->_TCW )[2];
            if ( !current->_depth.empty() )
            {
                unsigned short d = current->_depth.ptr<ushort> ( int ( fea->_pixel[1] ) ) [int ( fea->_pixel[0] )];
                if ( d==0 || d>10000 ) {
                    continue;
                }
                // LOG(INFO)<<"real depth = "<<double(d)/1000.0<<", estimated = "<<fea->_depth<<",px="<<fea->_pixel.transpose()<<endl;
                // fea->_depth = double ( d ) /1000.0;
            }
        }
    }
    */
}


void LocalMapping::LocalBA( Frame* current )
{
    // ba::LocalBA( _local_keyframes, _local_map_points );
    ba::LocalBAG2O( _local_keyframes, _local_map_points );
    
    // update the observations 
    for ( Feature* fea: current->_features )
    {
        if ( fea->_bad )
            continue;
        if ( fea->_mappoint)
        {
            if ( fea->_mappoint->_bad )
            {
                fea->_bad = true;
                continue;
            }
            fea->_mappoint->_last_seen = current->_keyframe_id;
            fea->_depth = current->_camera->World2Camera( fea->_mappoint->_pos_world, current->_TCW )[2];
        }
    }
    
}

// 更新局部关键帧: _local_keyframes 
// 寻找与当前帧的地图点有较好共视关系的那些关键帧
void LocalMapping::UpdateLocalKeyframes ( Frame* current )
{
    _local_keyframes.clear();
    _local_keyframes.insert(current);
    map<Frame*,int> keyframeCounter;
    for ( Feature* fea: current->_features ) {
        if ( fea->_mappoint && fea->_mappoint->_bad ==false)
        {
            for ( auto& obs_map : fea->_mappoint->_obs ) {
                keyframeCounter[obs_map.second->_frame] ++ ;
            }
        }
    }
    
    if ( keyframeCounter.empty() )
        return; 
    int max = 0;
    Frame* kfmax = nullptr;
    LOG(INFO)<<"keyframe counter size = "<<keyframeCounter.size()<<endl;
    
    // 策略1：能观测到当前帧 mappoint 的关键帧作为局部关键帧
    // 计算共视观测最多的帧
    for( auto& keyframe_pair: keyframeCounter ) {
        if ( keyframe_pair.first->_bad ) 
            continue;
        if ( keyframe_pair.first == current )
            continue;
        if ( keyframe_pair.second > max ) {
            max = keyframe_pair.second;
            kfmax = keyframe_pair.first;
        }
        
        _local_keyframes.insert( keyframe_pair.first );
    }
    
    // 策略2：和上次结果中共视程度很高的关键帧也作为局部关键帧
    int cnt_local_keyframes;
    for ( Frame* frame: _local_keyframes ) {
        vector<Frame*> neighbour = frame->GetBestCovisibilityKeyframes(10);
        for ( Frame* neighbour_kf: neighbour ) {
            if ( neighbour_kf->_bad ) 
                continue;
            _local_keyframes.insert( neighbour_kf );
        }
    }
    
    if ( kfmax ) {
        current->_ref_keyframe = kfmax; 
        LOG(INFO)<<"current "<<current->_keyframe_id<<"'s ref = "<<kfmax->_keyframe_id<<endl;
    }
    
    LOG(INFO)<<"local key frames: "<<_local_keyframes.size()<<endl;
}

// 更新与当前帧有关的地图点
void LocalMapping::UpdateLocalMapPoints ( Frame* current )
{
    _local_map_points.clear();
    // 将局部关键帧中的地图点添加到局部地图中
    
    for ( Frame* frame: _local_keyframes ) 
    {
        for ( Feature* fea: frame->_features ) 
        {
            if ( fea->_mappoint && fea->_mappoint->_bad==false
                && current->InFrame( current->_camera->World2Pixel(fea->_mappoint->_pos_world, current->_TCW))
            )
            {
                _local_map_points.insert( fea->_mappoint );
            }
        }
    }
    LOG(INFO)<<"Local map points: "<<_local_map_points.size()<<endl;
}

void LocalMapping::Run() 
{
    // while (1) {
    // 调通之后改成多线程形式
    while ( _new_keyframes.size() != 0 ) {
        // 将关键帧插入 local keyframes 并处理共视关系
        ProcessNewKeyFrame(); 
        
        // 剔除不合格的 map points 
        MapPointCulling();
        
        // 通过相邻帧间的特征匹配新建一些地图点
        CreateNewMapPoints();
        
        // 没有新的关键帧要处理
        if ( _new_keyframes.empty() ) {
            // 再搜索当前关键帧以及相邻关键帧之间的匹配
            SearchInNeighbors(); 
        }
        
        if ( _new_keyframes.empty() ) {
            // Local Bundle Adjustment 
            LocalBA( _current_kf );
            
            // Keyframe Culling 
            // 删掉一些没必要存在的关键帧，减少计算量
            KeyFrameCulling();
        }
        
        // 把这个帧加到闭环检测队列中
    } 
    // TODO 添加多线程功能
    
    //}
    
}
    
void LocalMapping::ProcessNewKeyFrame()
{
    _current_kf = _new_keyframes.front();
    _new_keyframes.pop_front();
    
    // 更新关键帧之间的连接关系：covisibility 和 essential 
    // Essential 现在还没实现，因为主要在loop closure里用
    _current_kf->UpdateConnections();
}



void LocalMapping::MapPointCulling()
{
    auto it = _recent_mappoints.begin();
    int th_obs = 2;
    
    while( it!=_recent_mappoints.end() ) {
        if ( (*it)->_bad ) {
            it = _recent_mappoints.erase( it );
        } else if ( (*it)->GetFoundRatio() < 0.25 ) {
            // 观测程度不够
            (*it)->_bad = true; 
            LOG(INFO)<<(*it)->_id<<" is set to bad"<<endl;
            it = _recent_mappoints.erase( it );
        } else if ( _current_kf->_id - (*it)->_last_seen >= 2 && (*it)->_cnt_found <= th_obs ) {
            // 从2个帧前拿到，但观测数量太少
            (*it)->_bad = true; 
            LOG(INFO)<<(*it)->_id<<" is set to bad"<<endl;
            it = _recent_mappoints.erase( it );
        } else if ( _current_kf->_keyframe_id - (*it)->_last_seen >= 2 ) {
            // 从三个帧前拿到但被没有剔除，认为是较好的点，但不再追踪
            it = _recent_mappoints.erase( it );
        } else {
            it++;
        }
    }
}

void LocalMapping::CreateNewMapPoints()
{
    int nn = 20;
    vector<Frame*> neighbour_kf = _current_kf->GetBestCovisibilityKeyframes(nn);
    LOG(INFO)<<"Neighbour key-frames: "<<neighbour_kf.size()<<endl;
    
    Vector3d cam_current = _current_kf->GetCamCenter();
    // Eigen::Matrix3d K_inv = _current_kf->_camera->GetCameraMatrix().inverse();
    
    int cnt_new_mappoints = 0;
    int cnt_associate_mps = 0;
    
    vector<pair<Feature*, Feature*>> new_feature_pairs;
    
    // 计算当前关键帧和别的关键帧之间的特征匹配关系
    for ( size_t i=0; i<neighbour_kf.size(); i++ ) {
        new_feature_pairs.clear();
        
        Frame* f2 = neighbour_kf[i];
        Vector3d baseline = cam_current - f2->GetCamCenter();
        double mean_depth, min_depth; 
        f2->GetMeanAndMinDepth( mean_depth, min_depth );
        double ratio_baseline_meandepth = baseline.norm()/mean_depth;
        if ( ratio_baseline_meandepth < 0.01 ) // 特别远？
            continue;
        
        // 通过极线去计算匹配关系
        // Fundamental 矩阵
        SE3 T12 = _current_kf->_TCW * f2->_TCW.inverse(); 
        // F12 = K^{-T} t_12^x R_12 K^{-1} 
        // E12 = t_12^x R_12, epipolar constraint: y1^T t12^x R12 y2 
        Eigen::Matrix3d E12 = SO3::hat( T12.translation() )*T12.rotation_matrix();
        
        vector<pair<int, int>> matched_pairs; 
        Matcher matcher;
        
        LOG(INFO)<<"searching "<<_current_kf->_keyframe_id<<" with "<<f2->_keyframe_id<<endl;
        int matches = matcher.SearchForTriangulation( _current_kf, f2, E12, matched_pairs);
        LOG(INFO)<<"matches in creating map points: "<<matches<<endl;
        
        // 对每个匹配点进行三角化
        for ( int im=0; im<matches; im++ ) {
            
            int i1 = matched_pairs[im].first;
            int i2 = matched_pairs[im].second;
            
            Feature* fea1 = _current_kf->_features[i1];
            Feature* fea2 = f2->_features[i2];
            
            
            if ( fea2->_mappoint == nullptr && fea1->_mappoint==nullptr ) {
                // 两个帧中的点均未被三角化，那么新建地图点
                Vector3d pt1 = _current_kf->_camera->Pixel2Camera( fea1->_pixel );
                Vector3d pt1_world = _current_kf->_camera->Camera2World( pt1, _current_kf->_TCW ); 
                Vector3d pt2 = f2->_camera->Pixel2Camera( fea2->_pixel );
                Vector3d pt2_world = f2->_camera->Camera2World( pt2, f2->_TCW ); 
                
                double cos_para_rays = pt1.dot(pt2) / (pt1.norm()*pt2.norm()) ;
                if ( cos_para_rays >= 0.9998 ) // 两条线平行性太强，不好三角化
                    continue; 
                double depth1 = 0, depth2=0;
                
                // 粗略估计其深度
                bool ret = cvutils::DepthFromTriangulation( T12.inverse(), pt1, pt2, depth1, depth2 ); 
                
                if ( ret==false || depth1 <0 || depth2<0 )
                    continue; 
                
                // 根据粗略的深度，进行位置校正
                fea1->_depth = depth1;
                Vector2d px_curr = fea2->_pixel;
                int level = 0;
                ret = matcher.FindDirectProjection( _current_kf, f2, fea1, px_curr, level );
                if ( ret == false )
                    continue;
                
                // 此时px_curr是更准确的像素位置，用它来进一步校准深度
                fea2->_pixel = px_curr;
                pt2 = f2->_camera->Pixel2Camera( fea2->_pixel );
                ret = cvutils::DepthFromTriangulation( T12.inverse(), pt1, pt2, depth1, depth2 ); 
                if ( ret==false || depth1 <0 || depth2<0 )
                    continue; 
                
                // 计算三角化点的重投影误差
                Vector3d pt1_triangulated = pt1*depth1;
                Vector2d px2_reproj = f2->_camera->Camera2Pixel( T12.inverse()*pt1_triangulated );
                double reproj_error = ( px2_reproj - fea2->_pixel ).norm();
                
                if ( reproj_error > 5.991 ) { // 重投影太大
                    continue; 
                }
                // 什么是尺度连续性。。。算了先不管它
                
                // 终于可以生成地图点了
                MapPoint* mp = Memory::CreateMapPoint();
                mp->_first_seen = mp->_last_seen = _current_kf->_keyframe_id;
                mp->_obs[ _current_kf->_keyframe_id ] = fea1;
                mp->_obs[ f2->_keyframe_id ] = fea2;
                mp->_cnt_visible = 2;
                mp->_cnt_found = 2;
                
                // 世界坐标的位置，可能取两个观测的平均更好一些？
                mp->_pos_world = _current_kf->_camera->Camera2World( pt1*depth1, _current_kf->_TCW );
                
                fea1->_mappoint = mp;
                fea2->_mappoint = mp;
                
                fea1->_depth = depth1;
                fea2->_depth = depth2;
                
                fea1->_bad = fea2->_bad = false;
                
                LOG(INFO)<<"create map point "<<mp->_id<<", pos = "<<mp->_pos_world.transpose()<<endl;
                ushort d = _current_kf->_depth.ptr<ushort>( cvRound(fea1->_pixel[1]) )[cvRound(fea1->_pixel[0])];
                LOG(INFO)<<"Estimated depth = "<<depth1<<", real depth = "<<double(d)/1000.0f << endl;;
                
                _recent_mappoints.push_back( mp );
                _local_map_points.insert( mp );         // 直接加到local map point中
                
                new_feature_pairs.push_back( make_pair(fea1, fea2) );
                cnt_new_mappoints++;
            } 
            else if ( fea2->_mappoint && fea1->_mappoint==nullptr ) 
            {
                // 第2帧中的这个点已经三角化，得到的匹配是它的投影，那么计算重投影
                Vector2d px_reproj = _current_kf->_camera->World2Pixel( fea2->_mappoint->_pos_world, _current_kf->_TCW );
                double reproj_error = (px_reproj - fea1->_pixel ).norm();
                if ( reproj_error > 5.991 ) { // 重投影太大
                    // LOG(INFO) << "reproj error too large: "<<reproj_error<<endl;
                    continue; 
                }
                
                // 匹配到了一个已经三角化了的点，将自己的观测量设过去
                fea1->_mappoint = fea2->_mappoint;
                fea1->_depth = _current_kf->_camera->World2Camera( fea2->_mappoint->_pos_world, _current_kf->_TCW )[2];
                fea1->_mappoint->_obs[_current_kf->_keyframe_id] = fea1;
                // 等待LocalBA更新此地图点的位置
                cnt_associate_mps++;
            } 
            else if ( fea1->_mappoint && fea2->_mappoint==nullptr )
            {
                // 反过来，第1帧的这个点已经三角化，但第2帧还没有
                Vector2d px_reproj = f2->_camera->World2Pixel( fea1->_mappoint->_pos_world, f2->_TCW );
                double reproj_error = (px_reproj - fea2->_pixel ).norm();
                if ( reproj_error > 5.991 ) { // 重投影太大
                    // LOG(INFO) << "reproj error too large: "<<reproj_error<<endl;
                    continue; 
                }
                
                fea2->_mappoint = fea1->_mappoint;
                fea2->_depth = f2->_camera->World2Camera( fea2->_mappoint->_pos_world, f2->_TCW )[2];
                fea2->_mappoint->_obs[f2->_keyframe_id] = fea2;
                cnt_associate_mps++;
            }
            else 
            {
                // 两个点都已经被三角化
                if ( fea1->_mappoint != fea2->_mappoint )
                {
                    // 但不是同一个点
                    // LOG(INFO)<<"this is strange"<<endl;
                    // LOG(INFO)<<"map point "<<fea1->_mappoint->_id<<" pos = "<<fea1->_mappoint->_pos_world.transpose()<<endl;
                    // LOG(INFO)<<"map point "<<fea2->_mappoint->_id<<" pos = "<<fea2->_mappoint->_pos_world.transpose()<<endl;
                }
                continue;
            }
        }
        
        /*
        if ( new_feature_pairs.empty() == false )
        {
            Mat color1_show = _current_kf->_color.clone();
            Mat color2_show = f2->_color.clone();
            
            for ( auto& fea_pair: new_feature_pairs )
            {
                Feature* fea1 = fea_pair.first;
                Feature* fea2 = fea_pair.second;
                cv::circle( color1_show, 
                    cv::Point2f(fea1->_pixel[0], fea1->_pixel[1]), 
                    1, cv::Scalar(0,250,0), 2
                );
                
                cv::circle( color2_show, 
                    cv::Point2f(fea2->_pixel[0], fea2->_pixel[1]), 
                    1, cv::Scalar(0,250,0), 2
                );
            }
            imshow("point in current frame", color1_show );
            imshow("point in past frame", color2_show );
            cv::waitKey();
        }
        */
    }
    
    LOG(INFO) << "New map points: " << cnt_new_mappoints << ", associated points: "<<cnt_associate_mps<<endl;
}

void LocalMapping::SearchInNeighbors()
{
    int nn = 20;
}

// 剔除一些冗余的关键帧
void LocalMapping::KeyFrameCulling()
{
    // 原则：如果某个关键帧的地图点有90%以上，都能被其他关键帧看到，则认为此关键帧是冗余的
    // 总帧数太少时，不要做culling
    LOG(INFO)<<"doing keyframe culling"<<endl;
    if ( _local_keyframes.size() <= 5 )
        return;
    vector<Frame*> local_keyframes = _current_kf->GetBestCovisibilityKeyframes();
    for ( Frame* frame : local_keyframes ) {
        if ( frame->_id == 0 ) 
            continue;
        
        const int th_obs = 3; 
        int redundant_observations = 0; 
        int mappoints = 0;
        
        for( Feature* fea: frame->_features ) {
            if ( fea->_mappoint && fea->_mappoint->_bad==false )
            {
                int nobs = 0;
                mappoints++;
                if ( fea->_mappoint->_obs.size() > th_obs ) {
                    // 至少被三个帧看到
                    for ( auto& obs_mp : fea->_mappoint->_obs ) {
                        Frame* obs_frame = obs_mp.second->_frame;
                        if ( obs_frame == frame ) 
                            continue;
                        nobs++;
                    }
                }
                
                if ( nobs >= th_obs )
                    redundant_observations++;
                
            }
        }
        if ( redundant_observations > 0.9*mappoints )
            frame->_bad = true; 
    }
}
    
}