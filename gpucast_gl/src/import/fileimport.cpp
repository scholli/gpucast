/********************************************************************************
*
* Copyright (C) 2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : fileimport.cpp
*  project    : glpp
*  description:
*
********************************************************************************/
// header i/f
#include "gpucast/gl/import/fileimport.hpp"

// header, system
#include <iostream>
#include <exception>

#include <gpucast/gl/graph/group.hpp>

#include <gpucast/gl/import/obj.hpp>
#include <gpucast/gl/import/fileparser.hpp>

#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/log/trivial.hpp>


namespace gpucast { namespace gl {


  /////////////////////////////////////////////////////////////////////////////
  // filetype::impl_t holds attached grammars
  /////////////////////////////////////////////////////////////////////////////
  class fileimport::impl_t 
  {
  public :

    typedef std::shared_ptr<fileparser>                         parser_ptr_type;

    typedef std::unordered_map<std::string, parser_ptr_type>    extension_map_type;

  public:

    impl_t()
      : _supported_filetypes()
    {
      // add your grammar here
      _supported_filetypes[".obj"] = parser_ptr_type(new fileparser_obj);
      
    }

    parser_ptr_type const& 
    get_parser(std::string const& extension) const
    {
      extension_map_type::const_iterator parser = _supported_filetypes.find(extension);
      if (parser != _supported_filetypes.end()) 
      {
        return parser->second;
      } else {
        throw std::runtime_error("file extension " + extension + " is not supported.\n");
      }
    }


  private : 

    extension_map_type   _supported_filetypes;

  };
  /////////////////////////////////////////////////////////////////////////////




  /////////////////////////////////////////////////////////////////////////////
  fileimport::fileimport()
    : _impl(new impl_t)
  {}


  /////////////////////////////////////////////////////////////////////////////
  /* virtual */ fileimport::~fileimport()      
  {}


  /////////////////////////////////////////////////////////////////////////////
  std::shared_ptr<node> 
  fileimport::load(std::string const& filename) const
  {
    try {
      if (boost::filesystem::exists(filename)) 
      {
        return _parse(filename);
      } else {
        throw std::runtime_error(filename + " does not exist.\n");
      }
    } 
    catch (std::exception const& e)
    {
      BOOST_LOG_TRIVIAL(error) << e.what() << std::endl;
      return std::shared_ptr<node> (new group);
    }
  }


  /////////////////////////////////////////////////////////////////////////////
  std::shared_ptr<node>   
  fileimport::_parse ( std::string const& filename ) const
  {
    impl_t::parser_ptr_type parser = _impl->get_parser(boost::filesystem::extension(filename));
    return parser->parse(filename);
  }

} } // namespace gpucast / namespace gl
