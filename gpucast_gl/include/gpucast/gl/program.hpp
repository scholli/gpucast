/********************************************************************************
*
* Copyright (C) 2009 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : program.hpp
*  project    : glpp
*  description:
*
********************************************************************************/
#ifndef GPUCAST_GL_PROGRAM_HPP
#define GPUCAST_GL_PROGRAM_HPP

// header, system
#include <cassert>

#if WIN32
  #pragma warning(disable : 4275) // dll-interface
#endif

// header, project
#include <gpucast/gl/shader.hpp>

namespace gpucast { namespace gl {

class texturebuffer;
class shaderstoragebuffer;
class cubemap;
class texturearray;
class texture1d;
class texture2d;
class texture3d;

class GPUCAST_GL program
{
public : // c'tor / d'tor

  program();
  virtual ~program();

public : // methods

  void        add                        ( shader* shader );
  void        remove                     ( shader* shader );

  void        link                       ( );

  void        begin                      ( ) const;
  void        end                        ( ) const;

  GLuint      id                         ( ) const;
  std::string log                        ( ) const;

  void        set_uniform1f              ( char const* varname, GLfloat v0 ) const;
  void        set_uniform2f              ( char const* varname, GLfloat v0, GLfloat v1 ) const;
  void        set_uniform3f              ( char const* varname, GLfloat v0, GLfloat v1, GLfloat v2 ) const;
  void        set_uniform4f              ( char const* varname, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3 ) const;

  void        set_uniform1i              ( char const* varname, GLint v0 ) const;
  void        set_uniform2i              ( char const* varname, GLint v0, GLint v1 ) const;
  void        set_uniform3i              ( char const* varname, GLint v0, GLint v1, GLint v2 ) const;
  void        set_uniform4i              ( char const* varname, GLint v0, GLint v1, GLint v2, GLint v3 ) const;

  void        set_uniform1fv             ( char const* varname, GLsizei count, GLfloat* value ) const;
  void        set_uniform2fv             ( char const* varname, GLsizei count, GLfloat* value ) const;
  void        set_uniform3fv             ( char const* varname, GLsizei count, GLfloat* value ) const;
  void        set_uniform4fv             ( char const* varname, GLsizei count, GLfloat* value ) const;

  void        set_uniform1iv             ( char const* varname, GLsizei count, GLint* value ) const;
  void        set_uniform2iv             ( char const* varname, GLsizei count, GLint* value ) const;
  void        set_uniform3iv             ( char const* varname, GLsizei count, GLint* value ) const;
  void        set_uniform4iv             ( char const* varname, GLsizei count, GLint* value ) const;

  void        set_uniform_matrix2fv      ( char const* varname, GLsizei count, GLboolean transpose, GLfloat* value ) const;
  void        set_uniform_matrix3fv      ( char const* varname, GLsizei count, GLboolean transpose, GLfloat* value ) const;
  void        set_uniform_matrix4fv      ( char const* varname, GLsizei count, GLboolean transpose, GLfloat const* value ) const;
  void        set_uniform_matrix2x3fv    ( char const* varname, GLsizei count, GLboolean transpose, GLfloat* value ) const;
  void        set_uniform_matrix2x4fv    ( char const* varname, GLsizei count, GLboolean transpose, GLfloat* value ) const;
  void        set_uniform_matrix3x2fv    ( char const* varname, GLsizei count, GLboolean transpose, GLfloat* value ) const;
  void        set_uniform_matrix3x4fv    ( char const* varname, GLsizei count, GLboolean transpose, GLfloat* value ) const;
  void        set_uniform_matrix4x2fv    ( char const* varname, GLsizei count, GLboolean transpose, GLfloat* value ) const;
  void        set_uniform_matrix4x3fv    ( char const* varname, GLsizei count, GLboolean transpose, GLfloat* value ) const;

  GLint       set_vertex_attrib          ( char const* varname ) const;

  void        set_texturebuffer          ( char const* varname, texturebuffer&, GLint unit ) const;
  void        set_shaderstoragebuffer    ( char const* varname, shaderstoragebuffer&, unsigned binding_point) const;

  void        set_texture1d              ( char const* varname, texture1d&, GLint unit ) const;
  void        set_texture2d              ( char const* varname, texture2d&, GLint unit ) const;
  void        set_texture3d              ( char const* varname, texture3d&, GLint unit ) const;
  void        set_texturearray           ( char const* varname, texturearray&, GLint unit ) const;
  void        set_cubemap                ( char const* varname, cubemap&, GLint unit ) const;

  // explicitly bind a variable name to a location before(!) linking the program
  void        bind_attrib_location       ( char const* varname, GLuint index ) const;
  void        bind_fragdata_location     ( char const* varname, GLuint index ) const;

  // get location of a variable after linking the program
  GLint       get_attrib_location        ( char const* varname ) const;
  GLint       get_uniform_location       ( char const* varname ) const;
  GLint       get_uniform_blockindex     ( char const* varname ) const;
  GLint       get_uniform_blocksize      ( char const* varname ) const;

  void        set_uniform_blockbinding   ( GLuint blockindex, GLuint blockbinding ) const;

  void        get_active_uniform_blockiv ( GLuint uniform_block_index, GLenum pname, GLint *params ) const;

private : // attributes

  program             (program const&);
  program& operator=  (program const&);

  GLuint      id_;
  GLint       is_linked_;

} ;

} } // namespace gpucast / namespace gl

#endif // GPUCAST_GL_PROGRAM_HPP

