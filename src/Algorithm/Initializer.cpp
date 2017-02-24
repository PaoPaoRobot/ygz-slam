#include "ygz/Algorithm/Initializer.h"

namespace ygz 
{
    
/**
 * @brief 并行地计算基础矩阵和单应性矩阵，选取其中一个模型，恢复出最开始两帧之间的相对姿态以及点云
 */
bool Initializer::TryInitialize(
    vector< Vector2d >& px1, 
    vector< Vector2d >& px2, 
    Frame* ref, 
    Frame* curr
)
{
    assert(px1.size() == px2.size());
    _ref = ref; _curr=curr;
    _num_points = px1.size();
    _px1 = px1;
    _px2 = px2;
    Matrix3d K = ref->_camera->GetCameraMatrix();
    
    _inliers = vector<bool>( px1.size(), true );
    _set = vector< vector<size_t> >(_options._max_iter, vector<size_t>(8,0) ); // 被选中的点
    
    // 使用RANSAC求解初始化
    vector<size_t> allIndices;  // 所有匹配点的索引
    allIndices.reserve( px1.size() );
    vector<size_t> availableIndices; 
    for ( size_t i=0; i<px1.size(); i++ )
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
    float sh=0, sf=0;       // h和e的评分
    Matrix3d H,F;
    
    thread threadH( &Initializer::FindHomography, this, std::ref(inlierH), std::ref(sh), std::ref(H) ); 
    thread threadF( &Initializer::FindFundamental, this, std::ref(inlierF), std::ref(sf), std::ref(F) ); 
    
    threadH.join();
    threadF.join();
    
    // 调试阶段不要并行
    // FindHomography( inlierH, sh, H );
    // FindFundamental( inlierF, sf, F );
    
    // 评价E和H哪个更好
    float rh = sh/(sh+sf);
    
    Matrix3d R21;
    Vector3d t21;
    vector<Vector3d> p3D;
    vector<bool> triangulated;
    
    // rh>0.4 认为H更好，否则认为F更好
    bool ret = false;
    if ( rh>0.4 )
        ret = ReconstructH( inlierH, H, K, R21, t21, p3D, triangulated, _options._min_parallex, _options._min_triangulated_pts );
    else 
        ret = ReconstructF( inlierF, F, K, R21, t21, p3D, triangulated, _options._min_parallex, _options._min_triangulated_pts );
    _T21 = SE3(R21, t21);
    if ( ret == true )
    {
        _inliers = triangulated;
        _pts_triangulated = p3D;
        return true;
    }
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
    Normalize( _px1, pn1, T1 );
    Normalize( _px2, pn2, T2 );
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
            pn1i[j] = pn1[idx];
            pn2i[j] = pn2[idx];
        }
        
        // 从八个点算H
        // 结果应该是 p2 = H21*p1
        Matrix3d Hn = ComputeH21( pn1i, pn2i );
        H21i = T2inv*Hn*T1;
        H12i = H21i.inverse();
        
        // set score 
        currentScore = CheckHomography( H21i, H12i, currentInliers, _options._sigma );
        
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

// |x'|     | h1 h2 h3 ||x|
// |y'| = a | h4 h5 h6 ||y|  简写: x' = a H x, a为一个尺度因子
// |1 |     | h7 h8 h9 ||1|
// 使用DLT(direct linear tranform)求解该模型
// x' = a H x 
// ---> (x') 叉乘 (H x)  = 0
// ---> Ah = 0
// A = | 0  0  0 -x -y -1 xy' yy' y'|  h = | h1 h2 h3 h4 h5 h6 h7 h8 h9 |
//     |-x -y -1  0  0  0 xx' yx' x'|
// 通过SVD求解Ah = 0，A'A最小特征值对应的特征向量即为解

/**
 * @brief 从特征点匹配求homography（normalized DLT）
 * 
 * @param  vP1 归一化后的点, in reference frame
 * @param  vP2 归一化后的点, in current frame
 * @return     单应矩阵
 * @see        Multiple View Geometry in Computer Vision - Algorithm 4.2 p109
 */
Matrix3d Initializer::ComputeH21(
    const vector<Vector2d>& vP1, const vector<Vector2d>& vP2)
{
    Eigen::MatrixXd A(2*vP1.size(), 9);
    for ( size_t i=0; i<vP1.size(); i++ ) 
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
    
    Eigen::JacobiSVD<Eigen::MatrixXd> svd ( A, Eigen::ComputeFullU|Eigen::ComputeFullV );
    Eigen::MatrixXd V = svd.matrixV();
    
    // Eigen的SVD是U*S*V^T = A ，所以取V最后一列
    
    Matrix3d ret; 
    ret<< V(0,8),V(1,8),V(2,8), 
          V(3,8),V(4,8),V(5,8), 
          V(6,8),V(7,8),V(8,8); 
    return ret;       // V 最后一列
}

/**
 * @brief 对给定的homography matrix打分
 * 
 * @see
 * - Author's paper - IV. AUTOMATIC MAP INITIALIZATION （2）
 * - Multiple View Geometry in Computer Vision - symmetric transfer errors: 4.2.2 Geometric distance
 * - Multiple View Geometry in Computer Vision - model selection 4.7.1 RANSAC
 * 
 * 实际上算的是第二个图到第一个图的H重投影
 */
float Initializer::CheckHomography(const Matrix3d& H21, const Matrix3d& H12, vector< bool >& inliers, float sigma)
{
    const double h11 = H21(0,0);
    const double h12 = H21(0,1);
    const double h13 = H21(0,2);
    const double h21 = H21(1,0);
    const double h22 = H21(1,1);
    const double h23 = H21(1,2);
    const double h31 = H21(2,0);
    const double h32 = H21(2,1);
    const double h33 = H21(2,2);
    
    const double h11inv = H12(0,0);
    const double h12inv = H12(0,1);
    const double h13inv = H12(0,2);
    const double h21inv = H12(1,0);
    const double h22inv = H12(1,1);
    const double h23inv = H12(1,2);
    const double h31inv = H12(2,0);
    const double h32inv = H12(2,1);
    const double h33inv = H12(2,2);
    
    inliers.resize( _num_points );
    
    float score = 0;
    const float th = 5.991; 
    const float invSigmaSquare = 1.0/(sigma*sigma);
    
    for ( int i=0; i<_num_points; i++ )
    {
        bool in = true;
        const double u1 = _px1[i][0];
        const double v1 = _px1[i][1];
        const double u2 = _px2[i][0];
        const double v2 = _px2[i][1];
        
        // reprojection error in the first image 
        // 将图像2中的特征点单应到图像1中
        // |u1|   |h11inv h12inv h13inv||u2|
        // |v1| = |h21inv h22inv h23inv||v2|
        // |1 |   |h31inv h32inv h33inv||1 |
        const float w2in1inv = 1.0/(h31inv*u2+h32inv*v2+h33inv);
        const float u2in1 = (h11inv*u2+h12inv*v2+h13inv)*w2in1inv;
        const float v2in1 = (h21inv*u2+h22inv*v2+h23inv)*w2in1inv;
        
         // 计算重投影误差
        const float squareDist1 = (u1-u2in1)*(u1-u2in1)+(v1-v2in1)*(v1-v2in1);

        // 根据方差归一化误差
        const float chiSquare1 = squareDist1*invSigmaSquare;

        if(chiSquare1>th)
        {
            inliers[i] = false;
        }
        else
        {
            score += th - chiSquare1;
            inliers[i] = true;
        }
    }
    
    return score;
}

// H矩阵分解常见有两种方法：Faugeras SVD-based decomposition 和 Zhang SVD-based decomposition
// 参考文献：Motion and structure from motion in a piecewise plannar environment
// 这篇参考文献和下面的代码使用了Faugeras SVD-based decomposition算法
/**
 * @brief 从H恢复R t
 *
 * @see
 * - Faugeras et al, Motion and structure from motion in a piecewise planar environment. International Journal of Pattern Recognition and Artificial Intelligence, 1988.
 * - Deeper understanding of the homography decomposition for vision-based control
 */
bool Initializer::ReconstructH(
    vector< bool >& inliers, Matrix3d& H21, Matrix3d& K, 
    Matrix3d& R21, Vector3d& t21, vector<Vector3d>& vP3D, 
    vector< bool >& triangulated, float minParallax, int minTriangulated)
{
    int N =0;
    for ( bool in: inliers )
        if ( in ) N++;

    Matrix3d invK = K.inverse();
    Matrix3d A = invK * H21 * K;
    Eigen::JacobiSVD<Matrix3d> svd( A, Eigen::ComputeFullU|Eigen::ComputeFullV );
    
    Matrix3d V = svd.matrixV();
    Matrix3d U = svd.matrixU();
    
    Eigen::Vector3d sigma = svd.singularValues();
    double d1 = ( sigma[0] );
    double d2 = ( sigma[1] );
    double d3 = ( sigma[2] );
    double s = U.determinant() * V.determinant();
    
    if ( d1/d2 < 1.00001 || d2/d3<1.00001 )
    {
        // 这种情况不应出现
        LOG(WARNING) << "FATAL Homography Initialization: This motion case is not implemented or is degenerate. Try again. " << endl;
        return false;
    }
    
    vector<HomographyDecomposition> decomps;     // 八个分解情况
    decomps.reserve(8);
    
    float aux1 = sqrt((d1*d1-d2*d2)/(d1*d1-d3*d3));
    float aux3 = sqrt((d2*d2-d3*d3)/(d1*d1-d3*d3));
    float x1[] = {aux1,aux1,-aux1,-aux1};
    float x3[] = {aux3,-aux3,aux3,-aux3};

    //case d'=d2
    float aux_stheta = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1+d3)*d2);

    float ctheta = (d2*d2+d1*d3)/((d1+d3)*d2);
    float stheta[] = {aux_stheta, -aux_stheta, -aux_stheta, aux_stheta};
    
    // 计算旋转矩阵 R'
    //      | costheta      0   -sintheta|       | aux1|
    // Rp = |    0        1       0      |  tp = |  0  |
    //      | stheta  0    costheta      |       |-aux3|

    //      | ctheta      0    aux_stheta|       | aux1|
    // Rp = |    0        1       0      |  tp = |  0  |
    //      |-aux_stheta  0    ctheta    |       | aux3|

    //      | ctheta      0    aux_stheta|       |-aux1|
    // Rp = |    0        1       0      |  tp = |  0  |
    //      |-aux_stheta  0    ctheta    |       |-aux3|

    //      | ctheta      0   -aux_stheta|       |-aux1|
    // Rp = |    0        1       0      |  tp = |  0  |
    //      | aux_stheta  0    ctheta    |       | aux3|
    for(size_t i=0; i<4; i++)
    {
        // Eq 13
        HomographyDecomposition decomp;
        Matrix3d Rp = Matrix3d::Identity();
        Rp(0,0) = ctheta;
        Rp(0,2) = -stheta[i];
        Rp(2,0) = stheta[i];
        Rp(2,2) = ctheta;
        decomp.R = s*U*Rp*V.transpose();
        
        // Eq 14
        decomp.t[0] = x1[i];
        decomp.t[1] = 0.0;
        decomp.t[2] = -x3[i];
        decomp.t = decomp.t*(d1-d3);
        decomp.t = U*decomp.t;
        decomp.t = decomp.t/decomp.t.norm();

        Vector3d np;
        np[0] = x1[i];
        np[1] = 0;
        np[2] = x3[i];
        decomp.n = V * np;
        if ( decomp.n[2]<0 ) 
            decomp.n=-decomp.n;
        decomps.push_back(decomp);
    }
    
    float aux_sphi = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1-d3)*d2);
    float cphi = (d1*d3-d2*d2)/((d1-d3)*d2);
    float sphi[] = {aux_sphi, -aux_sphi, -aux_sphi, aux_sphi};
    // case d' < 0
    for(size_t i=0; i<4; i++)
    {
        // Eq 15
        HomographyDecomposition decomp;
        Matrix3d Rp = Matrix3d::Identity();
        Rp(0,0) = cphi;
        Rp(0,2) = sphi[i];
        Rp(1,1) = -1;
        Rp(2,0) = sphi[i];
        Rp(2,2) = -cphi;
        decomp.R = s*U*Rp*V.transpose();

        // Eq 16
        Vector3d tp;
        tp[0] = x1[i];
        tp[1] = 0;
        tp[2] = x3[i];
        tp = tp*(d1+d3);
        decomp.t = U*tp;
        decomp.t = decomp.t/decomp.t.norm();
        
        Vector3d np;
        np[0] = x1[i];
        np[1] = 0;
        np[2] = x3[i];
        decomp.n = V * np;
        
        if ( decomp.n[2]<0 )
            decomp.n = -decomp.n;

        decomps.push_back(decomp);
    }
    
    // 分解之
    int bestGood = 0;
    int secondBestGood = 0;    
    int bestSolutionIdx = -1;
    float bestParallax = -1;
    vector<Vector3d> bestP3D;
    vector<bool> bestTriangulated;
    
    // Instead of applying the visibility constraints proposed in the Faugeras' paper (which could fail for points seen with low parallax)
    // We reconstruct all hypotheses and check in terms of triangulated points and parallax
    // 这里没有按照原文的八选四、四选二、二选一，而是直接尝试三角化并统计哪个解最优
    for(size_t i=0; i<8; i++)
    {
        double parallaxi=0;
        vector<Vector3d> vP3Di;
        vector<bool> vbTriangulatedi;
        
        int nGood = CheckRT( decomps[i].R, decomps[i].t, inliers, K,vP3Di, 4.0*_options._sigma2, vbTriangulatedi, parallaxi);

        // 保留最优的和次优的
        if(nGood>bestGood)
        {
            secondBestGood = bestGood;
            bestGood = nGood;
            bestSolutionIdx = i;
            bestParallax = parallaxi;
            bestP3D = vP3Di;
            bestTriangulated = vbTriangulatedi;
        }
        else if(nGood>secondBestGood)
        {
            secondBestGood = nGood;
        }
    }
    
    // LOG(INFO) << "best R = \n"<<decomps[bestSolutionIdx].R<<endl;
    // LOG(INFO) << "best t = "<<decomps[bestSolutionIdx].t.transpose()<<endl;
    
    if ( secondBestGood<0.75*bestGood && bestParallax>=minParallax 
        && bestGood>minTriangulated && bestGood > _options.good_point_ratio_H*_num_points )
    {
        // 这个条件还是蛮苛刻的，特别是90％的特征点都必须正确这一项
        R21 = decomps[bestSolutionIdx].R;
        t21 = decomps[bestSolutionIdx].t;
        vP3D = bestP3D;
        triangulated = bestTriangulated;
        return true;
    }
    
    // 分解H出现歧义，或者好点太少，总之就是不满足上面那四个条件，退出
    return false;
    
}

int Initializer::CheckRT(
    const Matrix3d& R, const Vector3d& t, 
    vector< bool >& inliers, const Matrix3d& K, 
    vector< Vector3d >& p3D, float th2, 
    vector< bool >& good, double& parallax, bool check_reprojection_error )
{
    const double fx = K(0,0);
    const double fy = K(1,1);
    const double cx = K(0,2);
    const double cy = K(1,2);
    
    good = vector<bool>(_num_points, false );
    p3D.resize( _num_points );
    vector<float> vcosParallax;
    vcosParallax.reserve( _num_points );
    
    // 第1个相机的投影矩阵
    Eigen::Matrix<double, 3,4> P1;
    P1.setZero();
    P1.block<3,3>(0,0) = K;
    Vector3d O1(0,0,0);
    
    // 第2个相机的投影矩阵
    Eigen::Matrix<double, 3,4> P2;
    P2.block<3,3>(0,0) = R;
    P2.block<3,1>(0,3) = t;
    P2 = K*P2;
    Vector3d O2 = -R.transpose()*t;
    
    int cntGood=0;      // 好点的数量
    for ( int i=0; i<_num_points; i++ )
    {
        // if ( inliers[i] == false )
            // continue;
        Vector3d p3dC1; // C1 坐标系下的3D点
        Triangulate( _px1[i], _px2[i], P1, P2, p3dC1 );
        if ( !isfinite(p3dC1[0]) ) // check the result 
        {
            good[i] = false;
            inliers[i] = false;
            continue;
        }
        // check parallax 
        Vector3d normal1 = p3dC1 - O1;
        double dist1 = normal1.norm();
        
        Vector3d normal2 = p3dC1 - O2; 
        double dist2 = normal2.norm();
        
        double cosParallax = normal1.dot(normal2) / (dist1*dist2);
        if ( p3dC1[2]<0 && cosParallax<0.99998 )
        {
            inliers[i]=false;
            continue; 
        }
        
        Vector3d p3dC2 = R*p3dC1 + t;
        if ( p3dC2[2]<0 && cosParallax<0.99998 )
        {
            inliers[i]=false;
            continue;
        }
        
        // check reprojection error
        if ( check_reprojection_error )
        {
            // in image1 
            double im1x, im1y;
            double invZ1 = 1.0/p3dC1[2];
            im1x = fx*p3dC1[0]*invZ1 + cx;
            im1y = fy*p3dC1[1]*invZ1 + cy;
            double squareError1 = ( im1x-_px1[i][0] )*( im1x-_px1[i][0] ) + ( im1y-_px1[i][1] )*( im1y-_px1[i][1] );
            if ( squareError1 > th2 )
            {
                inliers[i] = false;
                continue;
            }
            
            // in image2
            double im2x, im2y;
            double invZ2 = 1.0/p3dC2[2];
            im2x = fx*p3dC2[0]*invZ2 + cx;
            im2y = fy*p3dC2[1]*invZ2 + cy;
            double squareError2 = ( im2x-_px2[i][0] )*( im2x-_px2[i][0] ) + ( im2y-_px2[i][1] )*( im2y-_px2[i][1] );
            if ( squareError2 > th2 )
            {
                inliers[i] = false;
                continue;
            }
        }
        
        vcosParallax.push_back( cosParallax );
        p3D[i] = p3dC1;
        cntGood++;
        
        if ( cosParallax<0.99998 )
        {
            inliers[i] = true;
            good[i] = true;
        }
    }
    
    if ( cntGood > 0 )
    {
        sort( vcosParallax.begin(), vcosParallax.end() );
        size_t idx = min(50, int(vcosParallax.size()-1));
        parallax = acos(vcosParallax[idx])*180/M_PI;
    }
    else
        parallax = 0;
    return cntGood;
}

// Trianularization: 已知匹配特征点对{x x'} 和 各自相机矩阵{P P'}, 估计三维点 X
// x' = P'X  x = PX
// 它们都属于 x = aPX模型
//                         |X|
// |x|     |p1 p2  p3  p4 ||Y|     |x|    |--p0--||.|
// |y| = a |p5 p6  p7  p8 ||Z| ===>|y| = a|--p1--||X|
// |z|     |p9 p10 p11 p12||1|     |z|    |--p2--||.|
// 采用DLT的方法：x叉乘PX = 0
// |yp2 -  p1|     |0|
// |p0 -  xp2| X = |0|
// |xp1 - yp0|     |0|
// 两个点:
// |yp2   -  p1  |     |0|
// |p0    -  xp2 | X = |0| ===> AX = 0
// |y'p2' -  p1' |     |0|
// |p0'   - x'p2'|     |0|
// 变成程序中的形式：
// |xp2  - p0 |     |0|
// |yp2  - p1 | X = |0| ===> AX = 0
// |x'p2'- p0'|     |0|
// |y'p2'- p1'|     |0|
/**
 * @brief 给定投影矩阵P1,P2和图像上的点kp1,kp2，从而恢复3D坐标
 *
 * @param kp1 特征点, in reference frame
 * @param kp2 特征点, in current frame
 * @param P1  投影矩阵P1
 * @param P2  投影矩阵P2
 * @param x3D 三维点
 * @see       Multiple View Geometry in Computer Vision - 12.2 Linear triangulation methods p312
 */
void Initializer::Triangulate(
    const Vector2d& kp1, const Vector2d& kp2, 
    const Eigen::Matrix< double, int(3), int(4) >& P1, 
    const Eigen::Matrix< double, int(3), int(4) >& P2, 
    Vector3d& x3D )
{
    Eigen::Matrix4d A;
    A.block<1,4>(0,0) = kp1[0]*P1.block<1,4>(2,0)-P1.block<1,4>(0,0);
    A.block<1,4>(1,0) = kp1[1]*P1.block<1,4>(2,0)-P1.block<1,4>(1,0);
    A.block<1,4>(2,0) = kp2[0]*P2.block<1,4>(2,0)-P2.block<1,4>(0,0);
    A.block<1,4>(3,0) = kp2[1]*P2.block<1,4>(2,0)-P2.block<1,4>(1,0);
    Eigen::JacobiSVD<Eigen::Matrix4d> svd( A, Eigen::ComputeFullU|Eigen::ComputeFullV );
    x3D = svd.matrixV().block<3,1>(0,3)/svd.matrixV()(3,3);
}

/**
 * @brief 计算基础矩阵
 *
 * 假设场景为非平面情况下通过前两帧求取Fundamental矩阵(current frame 2 到 reference frame 1),并得到该模型的评分
 * 流程和FindHomography相似
 */
void Initializer::FindFundamental(
    vector<bool>& vbInliers, float& score, Matrix3d& F21)
{
    // normalized points 
    vector<Vector2d> pn1, pn2; 
    Matrix3d T1, T2;
    Normalize( _px1, pn1, T1 );
    Normalize( _px2, pn2, T2 );
    Matrix3d T2t = T2.transpose();
    
    // 最佳的inliers与评分
    score = 0;
    vbInliers = vector<bool>( _num_points, false );
    
    // 迭代中的变量
    vector<Vector2d> pn1i(8);
    vector<Vector2d> pn2i(8);
    Matrix3d F21i; 
    vector<bool> currentInliers( _num_points, false );
    float currentScore=0;
    
    for ( int it=0; it<_options._max_iter; it++ ) 
    {
        // RANSAC最小集合
        for ( size_t j=0; j<8; j++ )
        {
            int idx = _set[it][j];
            pn1i[j] = pn1[idx];
            pn2i[j] = pn2[idx];
        }
        
        // 计算F矩阵
        Matrix3d Fn = ComputeF21( pn1i, pn2i );
        F21i = T2t*Fn*T1;
        
        // 这里应该是 pn2i^T Fn pn1i = 0 
        // 或者 _px2&T F21i _px1 = 0
        
        // set score 
        currentScore = CheckFundamental( F21i, currentInliers, _options._sigma );
        
        if ( currentScore > score )
        {
            F21 = F21i;
            vbInliers = currentInliers;
            score = currentScore; 
        }
    }
}

// x'Fx = 0 整理可得：Af = 0
// A = | x'x x'y x' y'x y'y y' x y 1 |, f = | f1 f2 f3 f4 f5 f6 f7 f8 f9 |
// 通过SVD求解Af = 0，A'A最小特征值对应的特征向量即为解
/**
 * @brief 从特征点匹配求fundamental matrix（normalized 8点法）
 * @param  vP1 归一化后的点, in reference frame
 * @param  vP2 归一化后的点, in current frame
 * @return     基础矩阵
 * @see        Multiple View Geometry in Computer Vision - Algorithm 11.1 p282 (中文版 p191)
 */
Matrix3d Initializer::ComputeF21( 
    const vector< Vector2d >& vP1, 
    const vector< Vector2d >& vP2 ) 
{
    Eigen::MatrixXd A(vP1.size(), 9);
    for ( size_t i=0; i<vP1.size(); i++ ) 
    {
        const double u1 = vP1[i][0];
        const double v1 = vP1[i][1];
        const double u2 = vP2[i][0];
        const double v2 = vP2[i][1];
        
        A(i,0) = u2*u1;
        A(i,1) = u2*v1;
        A(i,2) = u2;
        A(i,3) = v2*u1;
        A(i,4) = v2*v1;
        A(i,5) = v2;
        A(i,6) = u1;
        A(i,7) = v1;
        A(i,8) = 1;
    }
    
    Eigen::JacobiSVD<Eigen::MatrixXd> svd ( A, Eigen::ComputeFullU|Eigen::ComputeFullV );
    Eigen::MatrixXd V = svd.matrixV();
    Matrix3d Fpre;
    Fpre << V(0,8),V(1,8),V(2,8), 
            V(3,8),V(4,8),V(5,8), 
            V(6,8),V(7,8),V(8,8);       // 最后一列转成3x3
    Eigen::JacobiSVD<Eigen::Matrix3d> svd_F ( Fpre, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Vector3d sigma = svd_F.singularValues();
    return svd_F.matrixU()*Eigen::DiagonalMatrix<double,3>(sigma[0], sigma[1],0)*svd_F.matrixV().transpose(); // 第3个奇异值设零
}

/**
 * @brief 对给定的fundamental matrix打分
 * 
 * @see
 * - Author's paper - IV. AUTOMATIC MAP INITIALIZATION （2）
 * - Multiple View Geometry in Computer Vision - symmetric transfer errors: 4.2.2 Geometric distance
 * - Multiple View Geometry in Computer Vision - model selection 4.7.1 RANSAC
 */
float Initializer::CheckFundamental(
    const Matrix3d& F21, vector< bool >& vbMatchesInliers, float sigma)
{
    const float f11 = F21(0,0);
    const float f12 = F21(0,1);
    const float f13 = F21(0,2);
    const float f21 = F21(1,0);
    const float f22 = F21(1,1);
    const float f23 = F21(1,2);
    const float f31 = F21(2,0);
    const float f32 = F21(2,1);
    const float f33 = F21(2,2);
    
    vbMatchesInliers.resize( _num_points );
    float score =0;
    
    // 基于卡方检验计算出的阈值（假设测量有一个像素的偏差）
    const float th = 3.841;
    const float thScore = 5.991;

    const float invSigmaSquare = 1.0/(sigma*sigma);

    for ( int i=0; i<_num_points; i++ ) 
    {
        bool bIn = true;
        const float u1 = _px1[i][0];
        const float v1 = _px1[i][1];
        const float u2 = _px2[i][0];
        const float v2 = _px2[i][1];

        // Reprojection error in second image
        // l2=F21x1=(a2,b2,c2)
        // F21x1可以算出x1在图像中x2对应的线l
        const float a2 = f11*u1+f12*v1+f13;
        const float b2 = f21*u1+f22*v1+f23;
        const float c2 = f31*u1+f32*v1+f33;

        // x2应该在l这条线上:x2点乘l = 0 
        const float num2 = a2*u2+b2*v2+c2;

        const float squareDist1 = num2*num2/(a2*a2+b2*b2); // 点到线的几何距离 的平方

        const float chiSquare1 = squareDist1*invSigmaSquare;

        if(chiSquare1>th)
            bIn = false;
        else
            score += thScore - chiSquare1;

        // Reprojection error in second image
        // l1 =x2tF21=(a1,b1,c1)

        const float a1 = f11*u2+f21*v2+f31;
        const float b1 = f12*u2+f22*v2+f32;
        const float c1 = f13*u2+f23*v2+f33;

        const float num1 = a1*u1+b1*v1+c1;

        const float squareDist2 = num1*num1/(a1*a1+b1*b1);

        const float chiSquare2 = squareDist2*invSigmaSquare;

        if(chiSquare2>th)
            bIn = false;
        else
            score += thScore - chiSquare2;

        if(bIn)
            vbMatchesInliers[i]=true;
        else
            vbMatchesInliers[i]=false;
    }
    return score;
}

bool Initializer::ReconstructF(
    vector< bool >& inliers, Matrix3d& F21, Matrix3d& K, 
    Matrix3d& R21, Vector3d& t21, vector< Vector3d >& vP3D, 
    vector< bool >& vbTriangulated, float minParallax, int minTriangulated)
{
    int N=0;
    for ( bool in: inliers )
        if ( in ) N++;
        
    Matrix3d E21 = K.transpose()*F21*K;
    Matrix3d R1, R2;
    Vector3d t;
    
    // 这里只有两个R，对t取正负号得到另两个解，共四个解
    DecomposeE( E21, R1, R2, t );
    
    Vector3d t1 = t;
    Vector3d t2 = -t;
    
    // 从四个解中选出最好的
    // 这代码谁写的。。。数组都不会用吗
    vector<Vector3d> p3D1, p3D2, p3D3, p3D4;
    vector<bool> triangulated1, triangulated2, triangulated3, triangulated4;
    double parallax1, parallax2, parallax3, parallax4; 
    
    // 注意F的分解容易受噪声影响，所以这里不妨把重投影的那个误差阈值调高一些(但是调太高会出现过多的similar)
    // 建议在检测F的时候不要用重投影，交给后面的BA来优化
    int good1 = CheckRT( R1, t1, inliers,K, p3D1, 24.0*_options._sigma2, triangulated1, parallax1, false ); 
    int good2 = CheckRT( R2, t1, inliers,K, p3D2, 24.0*_options._sigma2, triangulated2, parallax2, false ); 
    int good3 = CheckRT( R1, t2, inliers,K, p3D3, 24.0*_options._sigma2, triangulated3, parallax3, false ); 
    int good4 = CheckRT( R2, t2, inliers,K, p3D4, 24.0*_options._sigma2, triangulated4, parallax4, false ); 
    
    int maxGood = max(good1, max(good2, max(good3,good4)));
    int minGood = max( int(0.9*N), minTriangulated );
    int similar = 0;
    if ( good1>0.7*maxGood )
        similar++;
    if ( good2>0.7*maxGood )
        similar++;
    if ( good3>0.7*maxGood )
        similar++;
    if ( good4>0.7*maxGood )
        similar++;

    // 有歧义
    if ( maxGood<minGood || similar >1 )
        return false;
    
        // 比较大的视差角
    if(maxGood==good1)
    {
        if(parallax1>minParallax)
        {
            vP3D = p3D1;
            vbTriangulated = triangulated1;
            R21 = R1;
            t21 = t1;
            return true;
        }
    }else if(maxGood==good2)
    {
        if(parallax2>minParallax)
        {
            vP3D = p3D2;
            vbTriangulated = triangulated2;
            R21 = R2;
            t21 = t1;
            return true;
        }
    }else if(maxGood==good3)
    {
        if(parallax3>minParallax)
        {
            vP3D = p3D3;
            vbTriangulated = triangulated3;
            R21 = R1;
            t21 = t2;
            return true;
        }
    }else if(maxGood==good4)
    {
        if(parallax4>minParallax)
        {
            vP3D = p3D4;
            vbTriangulated = triangulated4;
            R21 = R2;
            t21 = t2;
            return true;
        }
    }
    return false;
}

void Initializer::DecomposeE(
    const Matrix3d& E, 
    Matrix3d& R1, Matrix3d& R2, 
    Vector3d& t)
{
    Eigen::JacobiSVD<Matrix3d> svd(E, Eigen::ComputeFullU|Eigen::ComputeFullV);
    Matrix3d U = svd.matrixU(), V = svd.matrixV();
    t = U.block<3,1>(0,2);
    t = t/t.norm();
    
    Matrix3d W;
    W.setZero();
    W(0,1) = -1;
    W(1,0) = 1;
    W(2,2) = 1;
    
    R1 = U*W*V.transpose();
    if ( R1.determinant() < 0 )
        R1 = -R1; 
    
    R2 = U*W.transpose()*V.transpose();
    if ( R2.determinant() < 0 )
        R2 = -R2;
}
    
}
