#include "ygz/Algorithm/Initializer.h"

namespace ygz 
{
    
/**
 * @brief 并行地计算基础矩阵和单应性矩阵，选取其中一个模型，恢复出最开始两帧之间的相对姿态以及点云
 */
bool Initializer::TryInitialize(
    const vector< Vector2d >& px1, 
    const vector< Vector2d >& px2, 
    Frame* ref, 
    Frame* curr
)
{
    assert(px1.size() == px2.size());
    _ref = ref; _curr=curr;
    _num_points = px1.size();
    _px1 = &px1;
    _px2 = &px2;
    
    _inliers = vector<bool>( px1.size(), true );
    _set = vector< vector<size_t> >(_options._max_iter, vector<size_t>(8,0) );
    
    // 使用RANSAC求解初始化
    vector<size_t> allIndices;  // 所有匹配点的索引
    allIndices.reserve( px1.size() );
    vector<size_t> availableIndices; 
    for ( int i=0; i<px1.size(); i++ )
        allIndices.push_back( i );
    
    cv::RNG rng;
    for ( int it=0; it<_options._max_iter; it++ )
    {
        availableIndices = allIndices;
        for ( int j=0; j<8; j++ )
        {
            // 从avaliable中随机选一个数作为index
            int rand = rng.uniform( 0, availableIndices.size() );
            int idx = availableIndices[rand];
            _set[it][j] = idx; 
            
            // 把选过的数删掉，当然不能在vector里用erase啦
            availableIndices[rand] = availableIndices.back();
            availableIndices.pop_back();
        }
    }
    
    // 对于选出的八对匹配，分别计算F和H
    vector<bool> inlierH, inlierF; 
    float sh, sf;       // h和e的评分
    Matrix3d H,F;
    
    thread threadH( &Initializer::FindHomography, this, ref(inlierH), ref(sh), ref(H) ); 
    thread threadF( &Initializer::FindFundamental, this, ref(inlierF), ref(sf), ref(F) ); 
    
    threadH.join();
    threadF.join();
    
    // 评价E和H哪个更好
    float rh = sh/(sh+sf);
    // rh>0.4 认为
    if ( rh>0.4 )
        return ReconstructH();
    else 
        return ReconstructF();
    return false;
}

void Initializer::FindHomography(
    vector< bool >& vbMatchesInliers, 
    float& score, Matrix3d& H21
)
{
    // normalized points 
    vector<Vector2d> pn1, pn2; 
    Matrix3d T1, T2;
    Normalize( *_px1, pn1, T1 );
    Normalize( *_px2, pn2, T2 );
    Matrix3d T2inv = T2.inverse();
    
    // 最佳的inliers与评分
    score = 0;
    vbMatchesInliers = vector<bool>( _num_points, false );
    // 迭代中的变量
    vector<Vector2d> pn1i(8);
    vector<Vector2d> pn2i(8);
    Matrix3d H21i, H12i; 
    vector<bool> currentInliers( _num_points, false );
    float currentScore=0;
    
    for ( int it=0; it<_options._max_iter; it++ ) 
    {
        // RANSAC最小集合
        for ( size_t j=0; j<8; j++ )
        {
            int idx = _set[it][j];
            pn1i = pn1[idx];
            pn2i = pn2[idx];
        }
        
        // 从八个点算H
        Matrix3d Hn = ComputeH21( pn1i, pn2i );
        H21i = T2inv*Hn*T1;
        H12i = H21i.inverse();
        
        // set score 
        currentScore = CheckHomography();
        
        if ( currentScore > score )
        {
            H21 = H21i;
            vbMatchesInliers = currentInliers;
            score = currentScore; 
        }
    }
    
}

void Initializer::Normalize(
    const vector< Vector2d >& pixels, 
    vector< Vector2d >& pixels_normalized, 
    Matrix3d& T )
{
    Vector2d mean; 
    for ( const Vector2d& px: pixels )
    {
        mean += px;
    }
    mean = mean/pixels.size();
    
    Vector2d meanDev;
    pixels_normalized.resize( pixels.size() );
    for ( size_t i=0; i<pixels.size(); i++ )
    {
        pixels_normalized[i] = pixels[i] - mean;
        meanDev[0] += fabs(pixels_normalized[i][0]);
        meanDev[1] += fabs(pixels_normalized[i][1]);
    }
    meanDev /= pixels.size();
    float sX = 1.0/meanDev[0];
    float sY = 1.0/meanDev[1];
    for ( size_t i=0; i<pixels.size(); i++ )
    {
        pixels_normalized[i][0] *= sX;
        pixels_normalized[i][1] *= sY;
    }
    
    // T =  |sX  0  -meanx*sX|
    //      |0   sY -meany*sY|
    //      |0   0      1    |
    T << sX,    0,      -mean[0]*sX,
          0,    sY,     -mean[1]*sY,
          0,    0,      1;
}

Mat Initializer::ComputeH21(
    const vector<Vector2d>& vP1, const vector<Vector2d>& vP2)
{
    Eigen::MatrixXd A(2*_num_points, 9);
    for ( int i=0; i<_num_points; i++ ) 
    {
        const double u1 = vP1[i][0];
        const double v1 = vP1[i][1];
        const double u2 = vP2[i][0];
        const double v2 = vP2[i][1];
        
        A(2*i,0) = 0.0;
        A(2*i,1) = 0.0;
        A(2*i,2) = 0.0;
        A(2*i,3) = -u1;
        A(2*i,4) = -v1;
        A(2*i,5) = -1;
        A(2*i,6) = v2*u1;
        A(2*i,7) = v2*v1;
        A(2*i,8) = v2;

        A(2*i+1,0) = u1;
        A(2*i+1,1) = v1;
        A(2*i+1,2) = 1;
        A(2*i+1,3) = 0.0;
        A(2*i+1,4) = 0.0;
        A(2*i+1,5) = 0.0;
        A(2*i+1,6) = -u2*u1;
        A(2*i+1,7) = -u2*v1;
        A(2*i+1,8) = -u2;
        
    }
    
    Eigen::JacobiSVD<>
    

}


    
}