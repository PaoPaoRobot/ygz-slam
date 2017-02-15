# 日志
## 17.2.15
- 重写了CeresAlignmentError.h。通过test/test_feature_projection进行测试。这个测试是先用DSO的pattern进行位姿估计，然后用Matcher::FindDirectProjection寻找地图点的投影
- Matcher又调用了CVUtils::Align2D进行块配准，其中又用了Ceres计算一个Affine warped的块匹配
- 结论：
Align的时候要选大一点的块，匹配才会准。SVO里选用64x64是有道理的。
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