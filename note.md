# 日志
## 17.2.25 
- 加入了Local BA机制。
- 加入了在Local mapping中对新增地图点进行检查的机制。然而现在local mapping的点还是太多了一些，在project之后可能会更多，比较影响效率。
- Depth估计是件比较麻烦的事情。在视线角非常接近时，地图点的深度值受像素误差影响严重，此时优化该地图点会导致观测不稳定。考虑引入DSO的Immature Points机制？

## 17.2.24
- 研究Local keyframes和local map points的合适数量。结论：在orb-slam中，local keyframes会不断增长至40至50左右，最高不超过80；在svo中，局部关键帧限制为10个。对于地图点，svo会移去离开视野的局部地图点，而orb则把共视的地图点全部加进来，可能有几千个点。同时，在orb的local mapping中，每个关键帧会新增100至150个地图点（默认每图1000特征点的情况下）。如果提高至2000个特征点，则新增地图点也会变多。
- LocalMapping.CreateNewMappoint的匹配应该没错，但是发现新增的地图点很难在前面的两次Track中用上。这可能是一个问题。
- 三角化当中可能存在一些问题。由于提特征点时用的网络滤波，所以即使两个特征点能够用描述子匹配上，它们的精度可能不足以确定其深度。需要进一步做实验验证。

## 17.2.23
- 测试LocalMapping中的逻辑。发现三角化点数量太少，于是检查了ORB描述的实现。然而ORB描述部分并没有太大问题，只是这种特征确实比较容易弄丢而已。
- 带bow之后匹配会变快，然而相对的，得到的匹配数量也会变少，因为首先要求这组匹配必须属于bow的同一个node。

## 17.2.22
- 添加了原生SVO的Align函数，见include/ygz/Algrithm/CVUtils.h。测试通过。
- VO的TrackRefFrame和TrackLocalMap终于可以跑起来了。

## 17.2.21
- 调试VO的track流程。
- 发现有时候Sparse Image Alignment会给出一个错误的结果，原因不明。可以对结果范数设阈值检查出来，但即使检查出来之后也似乎没有好办法来处理。
- 尝试将原生SVO中的两个alignment加进来。自己趟坑效率太低。
- 增加了原生SVO的sparse image alignment代码，见include/ygz/SparseImgAlign类。所需的依赖亦已添加。测试通过。 Release约2ms/200点，真是快。（日）
- 原有的matcher.SparseImageAlign在Release模式下运行会有错误，原因未查明。

## 17.2.20
- 调试VO中，测试了vo.trackRefFrame和TrackLocalMap的结果。现在工作正常但效果不好。
- 添加关键帧处理的代码。

## 17.2.19
- visual odometry和local mapping还在编写中。有不少工作要做。最好是每个重要算法都有测试。
- 测试了Matcher.SearchForTriangulation部分代码。它在BoW匹配基础上，检查了极线是否满足约束。在test/test_match_for_triangulation中测试了它的表现，通常在200个特征点中选出30左右的匹配（pose相近的话会更多一些）。
- 测试中发现Sparse Image Alignment在pose相差较大时似乎仍能工作，所以调高了检查结果合理性的那个阈值。但是它对深度噪声比较敏感。
- 增加了vo.init和vo.trackRefFrame，vo.TrackLocalMap的测试，见test目录下相应文件。 vo逻辑和svo差不多，一次track ref frame加一次track local map。接下来需要测试新增关键帧和地图点部分代码。

## 17.2.16
- 修复了initializer里，对F初始化时，对噪声太过敏感的问题。去除了由F计算R,t时的重投影检查。
- 开始写Visual Odometry部分。单目，关键帧特征点＋普通帧直接法。
- 测试了初始化后的双视图BA，程序加到test/test_initializer中。
- 结论：
Two View BA可能导致尺度发生变化，所以请在优化后重新归一尺度。
在有噪声的情况下，地图点越远，估计出来的结果越不准确。

## 17.2.15
- 重写了CeresAlignmentError.h。通过test/test_feature_projection进行测试。这个测试是先用DSO的pattern进行位姿估计，然后用Matcher::FindDirectProjection寻找地图点的投影
- Matcher又调用了CVUtils::Align2D进行块配准，其中又用了Ceres计算一个Affine warped的块匹配
- 结论：
Align的时候要选大一点的块，匹配才会准。SVO里选用64大小的patch是有道理的。
Ceres优化速度似乎比直接算GaussNewton慢一些？现在匹配200个点基本需要50多毫秒。
- 发现sparse alignment里居然忘了设pattern，一直用的一个点。。。现在改成了DSO那种八个点的pattern了，耗时约28ms/250个点。ceres真有效率问题


## 17.2.14
- 增加了ORB的计算以及基于BoW的匹配，见test/_test_orb_match.cpp。 
- 需要提供词典，词典文件见vocab/ORBvoc.bin 
- 结论：词典可以加速匹配（因为每个特征所属的单词是已知的，只需匹配一样的单词下的特征即可）。但是现在的特征提取部分用了grid，导致feature points重复性比较差，所以匹配数量不多。

- 开始测试2D align的问题，类似于光流的做法，但是如何用Ceres实现呢？

## 17.2.13 
- 增加了初始化部分代码，主要来自ORB-SLAM，见include/ygz/Initializer.h和src/Algorithm/Initializer.cpp文件。
- 测试文件为test/test_initializer.cpp，使用仿真数据。
- 结论：初始化H部分对噪声表现较好，但分解F的结果比较受噪声影响，需要增加一个后续的BA处理。

## 2017.2.11
- 调整Sparse Feature Alignment中的代码，将原有的8维误差改成了1维，提高了匹配速度. 3层金字塔，200个特征时，耗时约16ms。
- 算法见include/ygz/Ceres/CeresReprojSparseDirectError.h。由Matcher::SparseImageAlignment负责调用。
- 结论：
用FEJ会有少许速度提升，大约1到2ms，在alignment中并不明显。
深度值的误差会对alignment产生较大影响。

## 17.1 正在重构
- 小写开头的文件是旧文件，重构之后都以大写开头。
- SVO和ORB的基本元素已经加入到本程序中。