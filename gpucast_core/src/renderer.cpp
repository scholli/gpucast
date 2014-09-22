/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : renderer.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/

// header i/f
#include "gpucast/core/renderer.hpp"

// header, system
#include <fstream>
#include <regex>

#include <boost/tokenizer.hpp>
#include <boost/foreach.hpp>
#include <boost/filesystem.hpp>

// header, project

namespace gpucast {

  /////////////////////////////////////////////////////////////////////////////
  renderer::renderer()
    :  _width             ( 640 ),
       _height            ( 480 ),
       _pathlist          (),
       _background        ( 1.0f, 1.0f, 1.0f, 0.0f ),
       _nearplane         ( 1 ),
       _farplane          ( 10000 )
  {
    _pathlist.insert("");
    _pathlist.insert("./");
  }


  /////////////////////////////////////////////////////////////////////////////
  renderer::~renderer()
  {}


  /////////////////////////////////////////////////////////////////////////////
  void 
  renderer::add_path( std::string const& path, std::string separator)
  { 
    boost::char_separator<char> char_separator(separator.c_str());
    boost::tokenizer<boost::char_separator<char> > tokens(path, char_separator);

    for (auto t: tokens)
    {
      _pathlist.insert(t);
    }
  }


  /////////////////////////////////////////////////////////////////////////////
  float
  renderer::nearplane () const
  {
    return _nearplane;
  }


  /////////////////////////////////////////////////////////////////////////////
  float
  renderer::farplane () const
  {
    return _farplane;
  }


  /////////////////////////////////////////////////////////////////////////////
  void                          
  renderer::nearplane ( float n )
  {
    _nearplane = n;
    projectionmatrix(gpucast::math::frustum(-1.0f, 1.0f, -1.0f, 1.0f, _nearplane, _farplane));
  }


  /////////////////////////////////////////////////////////////////////////////
  void                          
  renderer::farplane ( float f )
  {
    _farplane = f;
    projectionmatrix(gpucast::math::frustum(-1.0f, 1.0f, -1.0f, 1.0f, _nearplane, _farplane));
  }


  /////////////////////////////////////////////////////////////////////////////
  void
  renderer::modelviewmatrix ( gpucast::math::matrix4f const& m )
  {
    // set matrix
    _modelviewmatrix = m;

    // recompute modelviewinverse
    _modelviewmatrixinverse = m;
    _modelviewmatrixinverse.invert();

    // recompute matrices that depend on modelviewmatrix
    _normalmatrix = m.normalmatrix();
    _modelviewprojectionmatrix = _projectionmatrix * _modelviewmatrix;
  }


  /////////////////////////////////////////////////////////////////////////////
  void
  renderer::projectionmatrix ( gpucast::math::matrix4f const& m )
  {
    // set matrix
    _projectionmatrix = m;

    // recompute matrices that depend on projection
    _modelviewprojectionmatrix        = _projectionmatrix * _modelviewmatrix;
    _modelviewprojectionmatrixinverse = gpucast::math::inverse(_modelviewprojectionmatrix);
  }


  /////////////////////////////////////////////////////////////////////////////
  gpucast::math::matrix4f const& 
  renderer::modelviewmatrix () const 
  {
    return _modelviewmatrix;
  }


  /////////////////////////////////////////////////////////////////////////////
  gpucast::math::matrix4f const& 
  renderer::modelviewmatrixinverse () const 
  {
    return _modelviewmatrixinverse;
  }


  /////////////////////////////////////////////////////////////////////////////
  gpucast::math::matrix4f const& 
  renderer::modelviewprojection () const 
  {
    return _modelviewprojectionmatrix;
  }


  /////////////////////////////////////////////////////////////////////////////
  gpucast::math::matrix4f const& 
  renderer::modelviewprojectioninverse () const 
  {
    return _modelviewprojectionmatrixinverse;
  }


  /////////////////////////////////////////////////////////////////////////////
  gpucast::math::matrix4f const& 
  renderer::normalmatrix () const 
  {
    return _normalmatrix;
  }


  /////////////////////////////////////////////////////////////////////////////
  gpucast::math::matrix4f const& 
  renderer::projectionmatrix () const 
  {
    return _projectionmatrix;
  }


  /////////////////////////////////////////////////////////////////////////////
  void                   
  renderer::background ( gpucast::math::vec4f const& rgba )
  {
    _background = rgba;
  }


  /////////////////////////////////////////////////////////////////////////////
  gpucast::math::vec4f const&     
  renderer::background ( ) const
  {
    return _background;
  }


  /////////////////////////////////////////////////////////////////////////////
  /* virtual */ void          
  renderer::resize ( int width, int height )
  {
    _width  = width;
    _height = height;
  }


  /////////////////////////////////////////////////////////////////////////////
  std::pair<bool, std::string>
  renderer::_path_to_file ( std::string const& filename ) const
  {
    std::set<std::string>::const_iterator path_to_source = _pathlist.begin();

    while (path_to_source != _pathlist.end())
    {
      if (boost::filesystem::exists( (*path_to_source) + filename ) ) {
        break;
      }

      ++path_to_source;
    }

    if ( path_to_source != _pathlist.end() ) 
    {
      std::fstream fstr (((*path_to_source) + filename).c_str(), std::ios::in);
      std::string source;
      if (fstr)
      {
        source = std::string((std::istreambuf_iterator<char>(fstr)), std::istreambuf_iterator<char>());

        std::smatch include_line;
        std::regex include_line_exp("#include[^\n]+");

        // as long as there are include lines
        while (std::regex_search(source, include_line, include_line_exp))
        {
          std::string include_line_str = include_line.str();
          std::smatch include_filename;
          std::regex include_filename_exp("[<\"][^>\"]+[>\"]");
          
          // extract include filename
          if (std::regex_search(include_line_str, include_filename, include_filename_exp))
          {
            std::string filename = include_filename.str();
            auto is_delimiter = [](char a) { return (a == '"') || (a == '<') || (a == '>'); };
            filename.erase(std::remove_if(filename.begin(), filename.end(), is_delimiter), filename.end());

            // replace include with source
            auto header_source = _path_to_file(filename);
            if (header_source.first) {
              source.replace(source.find(include_line.str()), include_line.str().size(), header_source.second);
            } else {
              source.replace(source.find(include_line.str()), include_line.str().size(), "");
              std::cerr << "renderer::_path_to_file(): Could not open " << filename << std::endl;
            }
          }
        }
      }
      fstr.close();

      return std::make_pair(true, source);
    } else {
      return std::make_pair(false, "");
    }
  }


#if 0
  /////////////////////////////////////////////////////////////////////////////
  void
  renderer::init_program ( std::shared_ptr<gpucast::gl::program>&  p,
                           std::string const&                 vertexshader_filename,
                           std::string const&                 fragmentshader_filename,
                           std::string const&                 geometryshader_filename )
  {
    try {
      gpucast::gl::vertexshader     vs;
      gpucast::gl::fragmentshader   fs;
      gpucast::gl::geometryshader   gs;

      p.reset ( new gpucast::gl::program );

      std::pair<bool, std::string> vspath = _path_to_file( vertexshader_filename );
      std::pair<bool, std::string> fspath = _path_to_file( fragmentshader_filename );
      std::pair<bool, std::string> gspath = _path_to_file( geometryshader_filename );

      if ( vspath.first  )
      {
        vs.set_source(vspath.second.c_str());
        vs.compile();
        if ( !vs.log().empty() ) {
          std::fstream ostr(boost::filesystem::basename(vertexshader_filename) + ".vert.log", std::ios::out);
          ostr << vs.log() << std::endl;
          ostr.close();
        }
        p->add(&vs);
      } else {
        throw std::runtime_error("renderer::_init_shader (): Couldn't open file " + vertexshader_filename);
      }

      if ( fspath.first  )
      {
        fs.set_source(fspath.second.c_str());
        fs.compile();
 
        if ( !fs.log().empty() ) {
          std::fstream ostr(boost::filesystem::basename(fragmentshader_filename) + ".frag.log", std::ios::out);
          ostr << vs.log() << std::endl;
          ostr.close();
        }
        p->add(&fs);
      } else {
        throw std::runtime_error("renderer::_init_shader (): Couldn't open file " + fragmentshader_filename);
      }

      if ( !gspath.first || geometryshader_filename.empty() )
      {
        // do nothing 
      } else {
        gs.set_source(gspath.second.c_str());
        gs.compile();
        if ( !gs.log().empty() ) {
          std::fstream ostr(boost::filesystem::basename(geometryshader_filename) + ".geom.log", std::ios::out);
          ostr << vs.log() << std::endl;
          ostr.close();
        }
        p->add(&gs);
      }

      // link all shaders
      p->link();

      if ( !p->log().empty() )
      {
        // stream log to std output
        std::cout << " program log : " << p->log() << std::endl;
      }
    } catch ( std::exception& e ) {
      std::cerr << "renderer::init_program(): failed to init program : " << vertexshader_filename << ", " << fragmentshader_filename << "( " << e.what () << ")\n";
    }
  }
#endif
} // namespace gpucast
