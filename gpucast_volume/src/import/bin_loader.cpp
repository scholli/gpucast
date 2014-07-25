/********************************************************************************
*
* Copyright (C) 2007-2012 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : bin_loader.cpp
*  project    : gpucast
*  description: binary volume loader
*
********************************************************************************/

// header i/f
#include "gpucast/volume/import/bin_loader.hpp"

// header, system
#include <fstream>
#include <stdexcept>

// header, project
#include <gpucast/volume/import/bin.hpp>
#include <gpucast/volume/beziervolumeobject.hpp>
#include <gpucast/volume/nurbsvolumeobject.hpp>

#include <boost/filesystem.hpp>


namespace gpucast {

///////////////////////////////////////////////////////////////////////
bin_loader::bin_loader()
{}

///////////////////////////////////////////////////////////////////////
bin_loader::~bin_loader()
{}

///////////////////////////////////////////////////////////////////////
bool 
bin_loader::load ( std::string const& filename, std::shared_ptr<beziervolumeobject> const& bptr )
{
  try {
    boost::filesystem::path inputfile ( filename );

    if ( !boost::filesystem::exists ( inputfile ) ) {
      throw std::runtime_error ( filename + " doesn't exists." );
    }

    std::string extension = boost::filesystem::extension ( inputfile );
    if ( extension != ".bin" ) {
      throw std::runtime_error ( "File not in binary format." );
    }

    std::fstream ifstr ( inputfile.string().c_str(), std::ios::in | std::ios_base::binary );
    bptr->read(ifstr);
    bptr->parent()->name(filename);

    ifstr.close();
    std::cerr << "Loading " << filename << " succeed." << std::endl;
    return true;
  } catch ( std::exception& e ) {
    std::cerr << "Loading " << filename << " failed. " << e.what() << std::endl;
    return false;
  }
}

} // namespace gpucast
