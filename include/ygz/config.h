#ifndef CONFIG_H_
#define CONFIG_H_

#include "ygz/common_include.h" 

namespace ygz 
{
class Config
{
public:
    ~Config();  // close the file when deconstructing 
    
    // set a new config file 
    static bool setParameterFile( const std::string& filename ); 
    
    // access the parameter values
    template< typename T >
    static T get( const std::string& key )
    {
        return T( Config::config_->file_[key] );
    }
    
private:
    static shared_ptr<Config> config_; 
    cv::FileStorage file_;
    
    Config () {} // private constructor makes a singleton
};
}

#endif // CONFIG_H_
