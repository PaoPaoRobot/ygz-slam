ygz-slam  一锅粥－SLAM

# 简介
- 一锅粥是一个特征点法结合直接法的视觉SLAM，为了便于研究，使用了模块化设计

# 依赖
- FAST
git clone https://github.com/uzh-rpg/fast.git

- Eigen
sudo apt-get install libeigen3-dev

- OpenCV 3.1 or higher 
see opencv.org

- Sophus 非模板类版本
    
    git clone https://github.com/strasdat/Sophus.git
    cd Sophus
    git checkout a621ff

- glog 
 sudo apt-get install libgoogle-glog-dev

- boost
 sudo apt-get install libboost-all-dev

- ceres solver
 https://github.com/ceres-solver/ceres-solver

- pangolin 
 https://github.com/stevenlovegrove/Pangolin

- DBoW3
 see 3rdparty

# 说明
## 一锅粥库由基础数据、算法、模块和应用四层组成
## 数据类见src/Basic，是组成SLAM系统的基本数据结构，例如帧、地图点等
## 算法类见src/Algorithm，操作基本数据构成算法。例如特征提取、匹配等。
## 模块调用算法，完成基本功能。如IO、VO、Local mapping等。
## 系统类组合使用各模块，构成SLAM系统，与最终用户交互。
