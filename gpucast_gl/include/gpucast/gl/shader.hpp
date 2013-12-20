/********************************************************************************
* 
* Copyright (C) 2009 Bauhaus-Universitaet Weimar                                               
*
*********************************************************************************
*
*  module     : shader.hpp                                        
*  project    : glpp 
*  description: 
*
********************************************************************************/
#ifndef GPUCAST_GL_SHADER_HPP
#define GPUCAST_GL_SHADER_HPP

#ifdef _MSC_VER
 #pragma warning(disable: 4996)
#endif

// header, system
#include <fstream>
#include <cassert>
#include <vector>

// header, project
#include <gpucast/gl/glpp.hpp>


namespace gpucast { namespace gl {

class GPUCAST_GL shader
{
protected :
  shader();

public :
  virtual       ~shader         ();

  void          load            ( std::string const& fileName );
  bool          compile         ( ) const;

  void          set_source      ( char const* source );
  std::string   get_source      ( ) const;  

  std::string   log             ( ) const;

  GLuint        id              ( );

protected :

  std::string   file_content     ( std::string const& fileName );
  void          replace_includes ( std::string& buffer );
  bool          find_include     ( std::string const& buffer, std::size_t& b, std::size_t& e );

protected :

  GLuint        id_;
};

} } // namespace gpucast / namespace gl gucast

#endif // GPUCAST_GL_SHADER_HPP
