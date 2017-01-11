#ifndef YGZ_CONFIG_H_
#define YGZ_CONFIG_H_

#include "ygz/Basic/Common.h" 


// Config 类
// 在程序运行前，用 Config::SetParameterFile() 设置配置文件
// 随后用 Get 获取参数的值
// 某些不需要在文件中配置的参数，亦定义在各类的 Option 结构中

namespace ygz 
{
class Config
{
public:
    ~Config();  // close the file when deconstructing 
    
    // set a new config file 
    static bool SetParameterFile( const std::string& filename ); 
    
    // access the parameter values
    template< typename T >
    static T Get( const std::string& key )
    {
        return T( Config::_config->_file[key] );
    }
    
private:
    static shared_ptr<Config> _config; 
    cv::FileStorage _file;
    
    Config () {} // private constructor makes a singleton
};
}

#endif // CONFIG_H_
