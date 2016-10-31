#include "ygz/config.h"

namespace ygz 
{
    
bool Config::setParameterFile( const std::string& filename )
{
    if ( config_ == nullptr )
        config_ = shared_ptr<Config>(new Config);
    config_->file_ = cv::FileStorage( filename.c_str(), cv::FileStorage::READ );
    if ( config_->file_.isOpened() == false )
    {
        LOG(FATAL)<<"parameter file "<<filename<<" does not exist."<<endl;
        config_->file_.release();
        return false;
    }
    return true;
}

Config::~Config()
{
    if ( file_.isOpened() )
        file_.release();
}

shared_ptr<Config> Config::config_ = nullptr;

}
