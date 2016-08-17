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
#include <memory>

// header, project
#include <gpucast/gl/glpp.hpp>


namespace gpucast { namespace gl {

enum shader_type {
  vertex_stage = 0x00,
  tesselation_control_stage,
  tesselation_evaluation_stage,
  geometry_stage,
  fragment_stage
};

struct GPUCAST_GL shader_desc {
  shader_type type;
  std::string filename;
};

class GPUCAST_GL shader
{
public :

  shader                        (shader_type type);
  shader                        (shader_type type, std::string const& filename);
  shader                        (shader_desc const& desc);
  ~shader                       ();

  void          reset           ();

  void          load            ( std::string const& filename );
  void          load            ( shader_type type, std::string const& filename);
  bool          compile         ( ) const;

  void          set_source      ( char const* source );
  std::string   get_source      ( ) const;  

  std::string   log             ( ) const;

  GLuint        id              ( ) const;
  shader_type   type            ( ) const;

private :

  void          _create           ( shader_type type );
  std::string   _file_content     ( std::string const& fileName );
  void          _replace_includes ( std::string& buffer );
  bool          _find_include     ( std::string const& buffer, std::size_t& b, std::size_t& e );

protected :

  GLuint        _id;
  shader_type   _type;
};

typedef std::shared_ptr<shader> shader_ptr;

} } // namespace gpucast / namespace gl gucast

#endif // GPUCAST_GL_SHADER_HPP
