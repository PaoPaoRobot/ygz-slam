#include <boost/timer.hpp>
#include "ygz/Basic.h"
#include "ygz/Algorithm/Matcher.h"
#include "ygz/Algorithm/CVUtils.h"
#include "ygz/CeresTypes.h"

namespace ygz {
    
Matcher::Matcher ()
{
    _options.th_low = Config::Get<int>("matcher.th_low");
    _options.th_high = Config::Get<int>("matcher.th_high");
    
    _options.init_low = Config::Get<int>("matcher.init_low");
    _options.init_high = Config::Get<int>("matcher.init_high");
    _options.knnRatio = Config::Get<int>("matcher.knnRatio");
}

Matcher::~Matcher()
{
    if ( !_patches_align.empty() ) {
        for ( uchar* p: _patches_align )
            delete[] p;
        _patches_align.clear();
    }
}

int Matcher::DescriptorDistance ( const Mat& a, const Mat& b )
{
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();
    int dist=0;
    for(int i=0; i<8; i++, pa++, pb++)
    {
        unsigned  int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }
    return dist;
}

int Matcher::CheckFrameDescriptors ( 
    Frame* frame1, 
    Frame* frame2, 
    list<pair<int,int>>& matches
)
{
    vector<int> distance; 
    for ( auto& m: matches )
    {
        distance.push_back( DescriptorDistance(
            frame1->_features[m.first]->_desc,
            frame2->_features[m.second]->_desc
        ));
    }
    
    int cnt_good = 0;
    int best_dist = *std::min_element( distance.begin(), distance.end() );
    LOG(INFO) << "best dist = "<<best_dist<<endl;
    
    // 取个上下限
    best_dist = best_dist>_options.init_low ? best_dist:_options.init_low; 
    best_dist = best_dist<_options.init_high ? best_dist:_options.init_high; 
    
    int i=0; 
    LOG(INFO)<<"original matches: "<<matches.size()<<endl;
    for ( auto iter=matches.begin(); iter!=matches.end(); i++ ) 
    {
        if ( distance[i] < _options.initMatchRatio*best_dist )  
        {
            cnt_good++;
            iter++;
        }
        else  
        {
            iter = matches.erase( iter );
        }
    }
    LOG(INFO) << "correct matches: "<<matches.size()<<endl;
    return cnt_good;
}

int Matcher::SearchForTriangulation ( 
    Frame* kf1, 
    Frame* kf2, 
    const Matrix3d& E12, 
    vector< pair< int, int > >& matched_points, 
    const bool& onlyStereo )
{
    DBoW3::FeatureVector& fv1 = kf1->_feature_vec;
    DBoW3::FeatureVector& fv2 = kf2->_feature_vec;
    
    LOG(INFO)<<fv1.size()<<","<<fv2.size()<<endl;
    assert( !fv1.empty() && !fv2.empty() );
    
    // 计算匹配
    int matches = 0;
    vector<bool> matched2( kf2->_features.size(), false ); 
    vector<int> matches12( kf1->_features.size(), -1 );
    vector<int> rotHist[ HISTO_LENGTH ];
    
    for ( int i=0; i<HISTO_LENGTH; i++ ) {
        rotHist[i].reserve(500);
    }
    const float factor = 1.0f/HISTO_LENGTH;
    
    // 将属于同一层的ORB进行匹配，利用字典加速
    DBoW3::FeatureVector::const_iterator f1it = fv1.begin();
    DBoW3::FeatureVector::const_iterator f2it = fv2.begin();
    DBoW3::FeatureVector::const_iterator f1end = fv1.end();
    DBoW3::FeatureVector::const_iterator f2end = fv2.end();
    
    while( f1it!=f1end && f2it!=f2end ) {
        if ( f1it->first == f2it->first ) {
            // 同属一个节点
            for ( size_t i1=0; i1<f1it->second.size(); i1++ ) {
                const size_t idx1 = f1it->second[i1]; 
                
                // 取出 kf1 中对应的特征点
                const Vector2d& kp1 = kf1->_features[idx1]->_pixel;
                const Mat& desp1 = kf1->_features[idx1]->_desc;
                
                int bestDist = 256;
                int bestIdx2 = -1;
                
                for ( size_t i2=0, iend2 = f2it->second.size(); i2<iend2; i2++ ) {
                    size_t idx2 = f2it->second[i2];
                    
                    Mat& desp2 = kf2->_features[idx2]->_desc;
                    
                    const int dist = DescriptorDistance( desp1, desp2 );
                    const Vector2d& kp2 = kf2->_features[idx2]->_pixel;
                    
                    if ( dist>_options.th_low || dist>bestDist ) 
                        continue;
                    
                    // 计算两个 keypoint 是否满足极线约束
                    Vector3d pt1 = kf1->_camera->Pixel2Camera( kp1 ); 
                    Vector3d pt2 = kf2->_camera->Pixel2Camera( kp2 ); 
                    if ( CheckDistEpipolarLine(pt1, pt2, E12) ) {
                        // 极线约束成立
                        bestIdx2 = idx2;
                        bestDist = dist;
                    } 
                    else 
                    {
                        LOG(INFO)<<"rejected by check epipolar line"<<endl;
                    }
                }
                
                if ( bestIdx2 >=0 ) {
                    matches12[idx1] = bestIdx2;
                    matches++;
                    
                    if ( _options.checkOrientation ) {
                        float rot = kf1->_features[idx1]->_angle - kf2->_features[bestIdx2]->_angle;
                        if ( rot<0 ) rot+=360;
                        int bin = round(rot*factor);
                        if ( bin == HISTO_LENGTH )
                            bin = 0;
                        assert( bin>=0 &&  bin<HISTO_LENGTH );
                        rotHist[bin].push_back( bestIdx2 );
                    }
                }
                
            }
            
            f1it++;
            f2it++;
        } else if ( f1it->first < f2it->first ) {
            f1it = fv1.lower_bound( f2it->first );
        } else {
            f2it = fv2.lower_bound( f1it->first );
        }
    }
    
    if ( _options.checkOrientation ) {
        // TODO 去掉旋转不对的点
    }
    
    matched_points.clear();
    matched_points.reserve( matches );
    
    for ( size_t i=0; i<matches12.size(); i++ ) {
        if ( matches12[i] >= 0 )
            matched_points.push_back( make_pair(i, matches12[i]) );
    }
    
    LOG(INFO) << "matches: "<<matches;
    return matches;
}

// 利用 Bag of Words 加速匹配
int Matcher::SearchByBoW(
    Frame* kf1, 
    Frame* kf2, 
    map<int,int>& matches )
{
    DBoW3::FeatureVector& fv1 = kf1->_feature_vec;
    DBoW3::FeatureVector& fv2 = kf2->_feature_vec;
    
    int cnt_matches = 0;
    
    vector<int> rotHist[HISTO_LENGTH]; // rotation 的统计直方图
    for ( int i=0;i <HISTO_LENGTH; i++ )
        rotHist[i].reserve(500);
    float factor = 1.0f/HISTO_LENGTH;
    
    DBoW3::FeatureVector::const_iterator f1it = fv1.begin();
    DBoW3::FeatureVector::const_iterator f2it = fv2.begin();
    DBoW3::FeatureVector::const_iterator f1end = fv1.end();
    DBoW3::FeatureVector::const_iterator f2end = fv2.end();
       
    while( f1it!=f1end && f2it!=f2end ) {
        if ( f1it->first == f2it->first ) {
            const vector<unsigned int> indices_f1 = f1it->second;
            const vector<unsigned int> indices_f2 = f2it->second;
            
            // 遍历 f1 中该 node 的特征点
            for ( size_t if1 = 0; if1<indices_f1.size(); if1++ ) {
                const unsigned int real_idx_f1 = indices_f1[if1];
                Mat desp_f1 = kf1->_features[real_idx_f1]->_desc;
                int bestDist1 = 256;  // 最好的距离
                int bestIdxF2 = -1;
                int bestDist2 = 256;  // 第二好的距离
                
                for ( size_t if2=0; if2<indices_f2.size(); if2++) {
                    const unsigned int real_idx_f2 = indices_f2[if2];
                    const Mat& desp_f2 = kf2->_features[real_idx_f2]->_desc;
                    const int dist = DescriptorDistance( desp_f1, desp_f2 );
                    if ( dist < bestDist1 ) {
                        bestDist2 = bestDist1;
                        bestDist1 = dist;
                        bestIdxF2 = real_idx_f2;
                    } else if ( dist<bestDist2 ) {
                        bestDist2 = dist; 
                    }
                }
                
                if ( bestDist1 < _options.th_low ) {
                    // 最小匹配距离小于阈值
                    if ( float(bestDist1) < _options.knnRatio*float(bestDist2) ) {
                        // 最好的匹配明显比第二好的匹配好
                        matches[ real_idx_f1 ] = bestIdxF2;
                        if ( _options.checkOrientation ) {
                            float rot = kf1->_features[real_idx_f1]->_angle - kf2->_features[bestIdxF2]->_angle;
                            if ( rot<0 ) rot+=360;
                            int bin = round(rot*factor);
                            if ( bin == HISTO_LENGTH )
                                bin = 0;
                            assert( bin>=0 &&  bin<HISTO_LENGTH );
                            rotHist[bin].push_back( bestIdxF2 );
                        }
                        cnt_matches++;
                    }
                }
            }
            
            f1it++;
            f2it++;
            
        } else if ( f1it->first < f2it->first ) {       // f1 iterator 比较小
            f1it = fv1.lower_bound( f2it->first );
        } else {        // f2 iterator 比较少
            f2it = fv2.lower_bound( f1it->first );
        }
    }
        
    if ( _options.checkOrientation ) {
        // 根据方向删除误匹配
        int ind1 = -1; 
        int ind2 = -1;
        int ind3 = -1;
        
        ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3 );
        
        for ( int i=0; i<HISTO_LENGTH; i++ ) {
            if ( i==ind1 || i==ind2 || i==ind3 )  // 保留之
                continue;
            for ( size_t j=0; j<rotHist[i].size(); j++ ) {
                rotHist[i][j];
                // TODO 删掉值为 rotHist[i][j] 的匹配
                
                cnt_matches--;
            }
        }
    }
    
    return cnt_matches;
}

void Matcher::ComputeThreeMaxima(
    vector<int>* histo, const int L, int& ind1, int& ind2, int& ind3)
{
    int max1=0;
    int max2=0;
    int max3=0;

    for(int i=0; i<L; i++)
    {
        const int s = histo[i].size();
        if(s>max1)
        {
            max3=max2;
            max2=max1;
            max1=s;
            ind3=ind2;
            ind2=ind1;
            ind1=i;
        }
        else if(s>max2)
        {
            max3=max2;
            max2=s;
            ind3=ind2;
            ind2=i;
        }
        else if(s>max3)
        {
            max3=s;
            ind3=i;
        }
    }

    if(max2<0.1f*(float)max1)
    {
        ind2=-1;
        ind3=-1;
    }
    else if(max3<0.1f*(float)max1)
    {
        ind3=-1;
    }
}

bool Matcher::CheckDistEpipolarLine(
    const Vector3d& pt1, const Vector3d& pt2, const Matrix3d& E12)
{
    const float a = pt1[0] * E12(0,0) + pt1[1] * E12(1,0) + E12(2,0);
    const float b = pt1[0] * E12(0,1) + pt1[1] * E12(1,1) + E12(2,1);
    const float c = pt1[0] * E12(0,2) + pt1[1] * E12(1,2) + E12(2,2);
    
    const float num = a * pt2[0] + b * pt2[1] + c;
    const float den = a*a+b*b;
    
    if ( den < 1e-6 )
        return false; 
    
    const float dsqr = num*num/den;
    
    return fabs(dsqr) < _options._epipolar_dsqr;
}

bool Matcher::FindDirectProjection(
    Frame* ref, Frame* curr, MapPoint* mp, Vector2d& px_curr, int& search_level)
{
    Eigen::Matrix2d ACR;
    Feature* fea = mp->_obs[ref->_keyframe_id];
    Vector2d& px_ref = fea->_pixel;
    Vector3d pt_ref = ref->_camera->Pixel2Camera( px_ref, fea->_depth );
    SE3 TCR = curr->_TCW*ref->_TCW.inverse();
    GetWarpAffineMatrix( ref, curr, px_ref, pt_ref, fea->_level, TCR, ACR );
    search_level = GetBestSearchLevel( ACR, curr->_option._pyramid_level-1 );
    WarpAffine( ACR, ref->_pyramid[fea->_level], fea->_pixel, fea->_level, search_level, WarpHalfPatchSize+1, _patch_with_border );
    // 去掉边界
    uint8_t* ref_patch_ptr = _patch;
    for ( int y=1; y<WarpPatchSize+1; ++y, ref_patch_ptr += WarpPatchSize )
    {
        uint8_t* ref_patch_border_ptr = _patch_with_border + y* ( WarpPatchSize+2 ) + 1;
        for ( int x=0; x<WarpPatchSize; ++x )
            ref_patch_ptr[x] = ref_patch_border_ptr[x];
    }
    Vector2d px_scaled = px_curr / (1<<search_level);
    bool success = cvutils::Align2DCeres( curr->_pyramid[search_level], _patch, _patch_with_border, px_scaled); 
    px_curr = px_scaled*(1<<search_level);
    if ( !curr->InFrame(px_curr) ) 
        return false;
    return success;
}

void Matcher::GetWarpAffineMatrix(
    const Frame* ref, const Frame* curr, 
    const Vector2d& px_ref, const Vector3d& pt_ref, 
    const int& level, const SE3& TCR, Matrix2d& ACR )
{
    Vector3d pt_ref_world = ref->_camera->Camera2World ( pt_ref, ref->_TCW );
    // 偏移之后的3d点，深度取成和pt_ref一致
    Vector3d pt_du_ref = ref->_camera->Pixel2Camera ( px_ref + Vector2d ( WarpHalfPatchSize, 0 ) * ( 1<<level ), pt_ref[2] );
    Vector3d pt_dv_ref = ref->_camera->Pixel2Camera ( px_ref + Vector2d ( 0, WarpHalfPatchSize ) * ( 1<<level ), pt_ref[2] );

    const Vector2d px_cur = curr->_camera->World2Pixel ( pt_ref_world, TCR );
    const Vector2d px_du = curr->_camera->World2Pixel ( pt_du_ref, TCR );
    const Vector2d px_dv = curr->_camera->World2Pixel ( pt_dv_ref, TCR );

    ACR.col ( 0 ) = ( px_du - px_cur ) / WarpHalfPatchSize;
    ACR.col ( 1 ) = ( px_dv - px_cur ) / WarpHalfPatchSize;
}

void Matcher::WarpAffine(
    const Matrix2d& ACR, const Mat& img_ref, const Vector2d& px_ref, 
    const int& level_ref, const int& search_level, 
    const int& half_patch_size, uint8_t* patch)
{
    const int patch_size = half_patch_size*2;
    const Eigen::Matrix2d ARC = ACR.inverse();

    // Affine warp
    uint8_t* patch_ptr = patch;
    const Vector2d px_ref_pyr = px_ref / ( 1<<level_ref );
    for ( int y=0; y<patch_size; y++ )
    {
        for ( int x=0; x<patch_size; x++, ++patch_ptr )
        {
            Vector2d px_patch ( x-half_patch_size, y-half_patch_size );
            px_patch *= ( 1<<search_level );
            const Vector2d px ( ARC*px_patch + px_ref_pyr );
            if ( px[0]<0 || px[1]<0 || px[0]>=img_ref.cols-1 || px[1]>=img_ref.rows-1 )
            {
                *patch_ptr = 0;
            }
            else
            {
                *patch_ptr = cvutils::GetBilateralInterpUchar ( px[0], px[1], img_ref );
            }
        }
    }
}

bool Matcher::SparseImageAlignment(Frame* ref, Frame* current)
{
    // from top to bottom 
    // _TCR_esti = SE3(); // reset estimation
    
    for ( int level = ref->_option._pyramid_level-1; level>=0; level-- ) 
    {
        SparseImageAlignmentInPyramid( ref, current, level );
    }
    
    if ( _TCR_esti.log().norm() > _options._max_alignment_motion ) {
        LOG(WARNING)<<"Too large motion: "<<_TCR_esti.log().norm()<< ". Reject this estimation. "<<endl;
        LOG(INFO)<<"TCR = \n"<<_TCR_esti.matrix()<<endl;
        _TCR_esti = SE3();
        return false;
    }
    
    return true;
}

bool Matcher::SparseImageAlignmentInPyramid(Frame* ref, Frame* current, int pyramid)
{
    PrecomputeReferencePatches( ref, pyramid );
    // solve the problem 
    ceres::Problem problem;
    Vector6d pose_curr;
    pose_curr.head<3>() = _TCR_esti.translation();
    pose_curr.tail<3>() = _TCR_esti.so3().log();
    // LOG(INFO)<<"start from "<<pose_curr.transpose()<<endl;
    
    int index = 0;
    for ( Feature* fea: ref->_features )
    {
        if ( fea->_mappoint && !fea->_mappoint->_bad && !fea->_bad )
        {
            problem.AddResidualBlock(
                new CeresReprojSparseDirectError(
                    ref->_pyramid[pyramid],
                    current->_pyramid[pyramid],
                    _patches_align[index++],
                    fea->_pixel,
                    ref->_camera->World2Camera( fea->_mappoint->_pos_world, ref->_TCW),
                    ref->_camera,
                    1<<pyramid,
                    _TCR_esti,
                    true
                ), 
                nullptr, 
                pose_curr.data()
            );
        }
    }
    
    ceres::Solver::Options options;
    options.num_threads = 2;
    options.num_linear_solver_threads =2;
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
    // options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve( options, &problem, &summary );
    
    // set the pose 
    _TCR_esti = SE3( 
        SO3::exp( pose_curr.tail<3>()),
        pose_curr.head<3>()
    );
    
    return true;
}


void Matcher::PrecomputeReferencePatches( Frame* ref, int level )
{
    // LOG(INFO) << "computing ref patches in level "<<level<<endl;
    if ( !_patches_align.empty() ) {
        for ( uchar* p: _patches_align )
            delete[] p;
        _patches_align.clear();
    }
    
    if (  !_duv_ref.empty() ) 
    {
        for ( Vector2d* d: _duv_ref )
            delete[] d;
        _duv_ref.clear();
    }
    
    boost::timer timer; 
    Mat& img = ref->_pyramid[level];
    int scale = 1<<level;
    for ( Feature* fea: ref->_features ) 
    {
        if ( fea->_mappoint && !fea->_mappoint->_bad )
        {
            Vector2d px_ref = fea->_pixel/scale;
            uchar * pixels = new uchar[PATTERN_SIZE];
            for ( int k=0; k<PATTERN_SIZE; k++ ) 
            {
                double u = px_ref[0]+PATTERN_DX[k];
                double v = px_ref[1]+PATTERN_DY[k];
                pixels[k] = cvutils::GetBilateralInterpUchar( u, v, img );
            }
            _patches_align.push_back( pixels );
        }
    }
    // LOG(INFO)<<"set "<<_patches_align.size()<<" patches."<<endl;
}



    
}