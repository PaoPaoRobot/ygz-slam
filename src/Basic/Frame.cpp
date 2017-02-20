#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "ygz/Basic/Frame.h"
#include "ygz/Basic/Camera.h"
#include "ygz/Basic/Memory.h"
#include "ygz/Basic/Config.h"
#include "ygz/Basic/Feature.h"
#include "ygz/Basic/MapPoint.h"
// #include "ygz/ORB/ORBVocabulary.h"

namespace ygz {
    
Frame::~Frame()
{
    if ( !_features.empty() )
    {
        CleanAllFeatures();
    }
}

void Frame::InitFrame()
{
    // convert the color into CV_8U grayscale 
    _pyramid.resize( _option._pyramid_level );
    Mat gray; 
    cv::cvtColor( _color, gray, CV_BGR2GRAY );
    _pyramid[0] = gray;
    CreateImagePyramid();
}

void Frame::CreateImagePyramid()
{
    for ( size_t i=1; i<_pyramid.size(); i++ ) 
    {
        // 在CV里使用down构造分辨率更低的图像，中间有高斯模糊化以降低噪声
        // 请记得第0层是原始分辨率
        cv::pyrDown( _pyramid[i-1], _pyramid[i] ); 
    }
}

bool Frame::GetMeanAndMinDepth ( double& mean_depth, double& min_depth )
{
    mean_depth = 0; min_depth = 9999;
    int cnt_valid_points = 0;
    for ( Feature* feature: _features ) 
    {
        if ( feature->_mappoint == nullptr )
            continue;
        if ( feature->_mappoint->_bad ) 
            continue;
        double depth = this->_camera->World2Camera( feature->_mappoint->_pos_world, this->_TCW )[2];
        if ( depth<0 ) 
            continue; 
        
        cnt_valid_points ++ ;
        mean_depth += depth;
        if ( depth < min_depth ) 
            min_depth = depth;
    }
    
    if ( cnt_valid_points == 0 ) 
    {
        mean_depth = 0; 
        min_depth = 0;
        return false; 
    }
    
    mean_depth /= cnt_valid_points;
    return true; 
}

std::vector< Frame* > Frame::GetBestCovisibilityKeyframes ( const int& N )
{
    if ( _cov_keyframes.size() < N ) 
        return _cov_keyframes;
    return vector<Frame*>( _cov_keyframes.begin(), _cov_keyframes.begin()+N );
}

bool Frame::IsInFrustum ( MapPoint* mp, float viewingCosLimit )
{
    // 判断是否在可见范围内
    return true;
}

void Frame::UpdateConnections()
{
    map<Frame*, int> kfCounter;
    for ( Feature* fea: _features ) 
    {
        MapPoint* mp = fea->_mappoint;
        if ( !mp || mp->_bad ) 
            continue;
        for ( auto& obs_mp: mp->_obs ) 
        {
            // 自己和自己不算共视
            if ( obs_mp.first == _keyframe_id )  
                continue; 
            kfCounter[ Memory::GetKeyFrame(obs_mp.first) ] ++;
        }
    }
    
    if ( kfCounter.empty() ) 
        return; 
    
    int nmax = 0;
    Frame* kfmax = nullptr;
    int th = 15;
    vector<pair<int, Frame*> > pairs; 
    pairs.reserve( kfCounter.size() );
    
    // 寻找共视程度最好的帧
    for ( auto& kf_pair: kfCounter ) 
    {
        if ( kf_pair.second > nmax ) 
        {
            nmax = kf_pair.second; 
            kfmax = kf_pair.first;
        }
        if ( kf_pair.second>=th ) 
        {
            // 共视点大于阈值
            pairs.push_back( make_pair(kf_pair.second, kf_pair.first) );
            // kf_pair.AddConnection(); 
        }
    }
    
    if ( pairs.empty() ) 
    {
        // 没有超过阈值的共视帧，以最大的为准
        pairs.push_back( make_pair(nmax, kfmax ));
        kfmax->AddConnection( this, nmax );
    }
    
    sort( pairs.begin(), pairs.end() );
    list<Frame*> lKFs;
    list<int> lWs;
    for ( size_t i=0; i<pairs.size(); i++ ) 
    {
        lKFs.push_front( pairs[i].second );
        lWs.push_front( pairs[i].first );
    }
    
    // 更新 cov 和 essential 
    _connected_keyframe_weights = kfCounter;
    _cov_keyframes = vector<Frame*> (lKFs.begin(), lKFs.end());
    _cov_weights = vector<int>( lWs.begin(), lWs.end() );
    
    LOG(INFO) << "convisible keyframes: "<<_cov_keyframes.size()<<endl;
    
    // Essential 待议
}

void Frame::AddConnection ( Frame* kf, const int& weight )
{
    if ( !_connected_keyframe_weights.count(kf) )
        _connected_keyframe_weights[kf] = weight;
    else
        _connected_keyframe_weights[kf] = weight;
}

void Frame::UpdateBestCovisibles()
{
    vector<pair<int, Frame*>> pairs; 
    pairs.reserve( _connected_keyframe_weights.size() );
    
    for ( auto& kf_weights: _connected_keyframe_weights ) 
        pairs.push_back( make_pair(kf_weights.second, kf_weights.first ) );
    sort( pairs.begin(), pairs.end() );
    
    for ( auto riter = pairs.rbegin(); riter!=pairs.rend(); riter++ ) 
    {
        _cov_keyframes.push_back( riter->second );
        _cov_weights.push_back( riter->first );
    }
}

Mat Frame::GetAllDescriptors()
{
    Mat alldesp( _features.size(), 32, CV_8U );
    int index = 0;
    for ( Feature* fea: _features ) 
    {
        fea->_desc.copyTo( alldesp.row(index) );
        index++;
    }
    return alldesp;
}

void Frame::ComputeBoW()
{
    if ( _vocab!=nullptr && _bow_vec.empty() ) 
    {
        vector<Mat> alldesp;
        for ( Feature* fea: _features ) 
        {
            alldesp.push_back(fea->_desc);
        }
        _vocab->transform( alldesp, _bow_vec, _feature_vec, 4);
    }
}

void Frame::CleanAllFeatures()
{
    for ( size_t i=0; i<_features.size(); i++ ) 
    {
        delete _features[i];
    }
    _features.clear();
}


PinholeCamera* Frame::_camera = nullptr;
ORBVocabulary* Frame::_vocab = nullptr;

}
