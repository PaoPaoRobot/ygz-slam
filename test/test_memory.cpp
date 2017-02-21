#include "ygz/Basic.h"
#include "ygz/Algorithm.h"

// 这个只是为了测试调试器会不会因为内存问题崩掉
int main( int argc, char** argv )
{
    int index = 0;
    while(1) 
    {
        ygz::Frame* newframe = new ygz::Frame();
        newframe->_color = cv::Mat(640,480,CV_8UC3);
        LOG(INFO)<<"create frame "<<index<<endl;
        index++;
    }
    return 0;
}