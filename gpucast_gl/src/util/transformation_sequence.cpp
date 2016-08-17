/********************************************************************************
*
* Copyright (C) 2012 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : transformation_sequence.cpp
*  project    : glpp
*  description:
*
********************************************************************************/
// include i/f header
#include "gpucast/gl/util/transformation_sequence.hpp"

#include <gpucast/gl/glpp.hpp>

#include <fstream>
#include <boost/filesystem.hpp>
#include <boost/log/trivial.hpp>

namespace gpucast { namespace gl {

  /////////////////////////////////////////////////////////////////////////////
  transformation_sequence::transformation_sequence  ()
    : _transformations()
  {}

  /////////////////////////////////////////////////////////////////////////////
  transformation_sequence::~transformation_sequence ()
  {}

  /////////////////////////////////////////////////////////////////////////////
  void                      
  transformation_sequence::add ( gpucast::math::matrix4f const& mv, gpucast::math::matrix4f const& mvi, gpucast::math::matrix4f const& mvp, gpucast::math::matrix4f const& mvpi, gpucast::math::matrix4f const& nm )
  {
    _transformations.push_back ( transformation_set (mv, mvi, mvp, mvpi, nm ) );
  }

  /////////////////////////////////////////////////////////////////////////////
  void                      
  transformation_sequence::clear ()
  {
    _transformations.clear();
  }

  /////////////////////////////////////////////////////////////////////////////
  bool                      
  transformation_sequence::empty () const
  {
    return _transformations.empty();
  }

  /////////////////////////////////////////////////////////////////////////////
  transformation_set const& 
  transformation_sequence::next () const
  {
    return _transformations.front();
  }

  /////////////////////////////////////////////////////////////////////////////
  void                      
  transformation_sequence::pop ()
  {
    _transformations.pop_front();
  }

  /////////////////////////////////////////////////////////////////////////////
  void                      
  transformation_sequence::write ( std::string const& file ) const
  {
    try {
      std::ofstream fstr ( file.c_str(), std::ios_base::binary | std::ios_base::out );
      if ( fstr.good() )
      {
        write ( fstr );
        fstr.close();
      } else {
        throw std::runtime_error("Cannot open file.");
      }
    } catch ( std::exception& e ) {
      BOOST_LOG_TRIVIAL(error) << "transformation_sequence::write() failed. " << e.what() << std::endl;
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  void                      
  transformation_sequence::read  ( std::string const& file )
  {
    try 
    {
      if ( boost::filesystem::exists ( file ) )
      {
        std::ifstream fstr ( file.c_str(), std::ios_base::binary | std::ios_base::in );
        if ( fstr.good() )
        {
          read ( fstr );
          fstr.close();
        } else {
          throw std::runtime_error("Cannot open file.");
        }
      } else {
        throw std::runtime_error("No such file.");
      }
    } catch ( std::exception& e ) {
      BOOST_LOG_TRIVIAL(error) << "transformation_sequence::read() failed. " << e.what() << std::endl;
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  void                      
  transformation_sequence::write ( std::ostream& os ) const
  {
    std::size_t size = _transformations.size();
    os.write ( reinterpret_cast<char const*> (&size), sizeof(std::size_t));
    std::for_each ( _transformations.begin(), _transformations.end(), std::bind(&transformation_set::write, std::placeholders::_1, std::ref(os)));
  }

  /////////////////////////////////////////////////////////////////////////////
  void                      
  transformation_sequence::read  ( std::istream& is )
  {
    std::size_t size = 0;
    is.read ( reinterpret_cast<char*> (&size), sizeof(std::size_t));
    _transformations.resize(size);
    std::for_each(_transformations.begin(), _transformations.end(), std::bind(&transformation_set::read, std::placeholders::_1, std::ref(is)));
  }

} } // namespace gpucast / namespace gl

