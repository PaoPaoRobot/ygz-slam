#include "ygz/Basic/Config.h"

namespace ygz 
{
    
bool Config::SetParameterFile( const std::string& filename )
{
    if ( _config == nullptr )
        _config = shared_ptr<Config>(new Config);
    _config->_file = cv::FileStorage( filename.c_str(), cv::FileStorage::READ );
    if ( _config->_file.isOpened() == false )
    {
        LOG(ERROR)<<"parameter file "<<filename<<" does not exist."<<endl;
        _config->_file.release();
        return false;
    }
    return true;
}

Config::~Config()
{
    if ( _file.isOpened() )
        _file.release();
}

shared_ptr<Config> Config::_config = nullptr;

}
