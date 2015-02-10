/********************************************************************************
*
* Copyright (C) 2009 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : program.cpp
*  project    : glpp
*  description:
*
********************************************************************************/

// header i/f
#include "gpucast/gl/program.hpp"

#include <gpucast/gl/texturebuffer.hpp>
#include <gpucast/gl/shaderstoragebuffer.hpp>
#include <gpucast/gl/texture1d.hpp>
#include <gpucast/gl/texture2d.hpp>
#include <gpucast/gl/texture3d.hpp>
#include <gpucast/gl/texturearray.hpp>
#include <gpucast/gl/cubemap.hpp>
#include <gpucast/gl/error.hpp>

// header, system
#include <iostream>
#include <GL/glew.h>



namespace gpucast { namespace gl {

////////////////////////////////////////////////////////////////////////////////
program::program()
{
  id_ = glCreateProgram();
}


////////////////////////////////////////////////////////////////////////////////
program::~program()
{
  glDeleteProgram(id_);
}


////////////////////////////////////////////////////////////////////////////////
void
program::add(shader* shader)
{
  glAttachShader(id_, shader->id());
}


////////////////////////////////////////////////////////////////////////////////
void
program::remove(shader* shader)
{
  glDetachShader(id_, shader->id());
}


////////////////////////////////////////////////////////////////////////////////
void
program::link()
{
  glLinkProgram(id_);

  glGetProgramiv(id_, GL_LINK_STATUS, &is_linked_);

  if (is_linked_ != GL_TRUE)
  {
    std::cerr << log() << std::endl;
  }

  //assert(is_linked_ && "program::link() : assertion when linking program");
}


////////////////////////////////////////////////////////////////////////////////
void
program::begin() const
{
  glUseProgram(id_);
}


////////////////////////////////////////////////////////////////////////////////
void
program::end() const
{
  glUseProgram(0);
}


////////////////////////////////////////////////////////////////////////////////
GLuint program::id() const
{
  return id_;
}


////////////////////////////////////////////////////////////////////////////////
std::string 
program::log() const
{
  GLint log_len;
  glGetProgramiv(id_, GL_INFO_LOG_LENGTH, &log_len);

  GLint size;
  char* buf = new char[log_len];

  glGetProgramInfoLog(id_, log_len, &size, buf);

  std::string infolog(buf);
  delete[] buf;

  return infolog;
}


////////////////////////////////////////////////////////////////////////////////
void
program::set_uniform1f(char const* varname, GLfloat v0) const
{
  GLint location = get_uniform_location(varname);

  if (location >= 0) 
  {
#if GPUCAST_GL_DIRECT_STATE_ACCESS
    glProgramUniform1fEXT(id_, location, v0);
#else 
    glUniform1f(location, v0);
#endif
  } 
}


////////////////////////////////////////////////////////////////////////////////
void
program::set_uniform2f(char const* varname, GLfloat v0, GLfloat v1) const
{
  GLint location = get_uniform_location(varname);

  if (location >= 0) 
  {
#if GPUCAST_GL_DIRECT_STATE_ACCESS
    glProgramUniform2fEXT(id_, location, v0, v1);
#else 
    glUniform2f(location, v0, v1);
#endif
  }
}


////////////////////////////////////////////////////////////////////////////////
void
program::set_uniform3f(char const* varname, GLfloat v0, GLfloat v1, GLfloat v2) const
{
  GLint location = get_uniform_location(varname);

  if (location >= 0) 
  {
#if GPUCAST_GL_DIRECT_STATE_ACCESS
    glProgramUniform3fEXT(id_, location, v0, v1, v2);
#else 
    glUniform3f(location, v0, v1, v2);
#endif
  }
}


////////////////////////////////////////////////////////////////////////////////
void
program::set_uniform4f(char const* varname, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3) const
{
  GLint location = get_uniform_location(varname);

  if (location >= 0) 
  {
#if GPUCAST_GL_DIRECT_STATE_ACCESS
    glProgramUniform4fEXT(id_, location, v0, v1, v2, v3);
#else 
    glUniform4f(location, v0, v1, v2, v3);
#endif
  } 

  //assert((location >= 0) && "set_uniform4f() : uniform not in use");
}


////////////////////////////////////////////////////////////////////////////////
void
program::set_uniform1i(char const* varname, GLint v0) const
{
  GLint location = get_uniform_location(varname);

  if (location >= 0) 
  {
#if GPUCAST_GL_DIRECT_STATE_ACCESS
    glProgramUniform1iEXT(id_, location, v0);
#else 
    glUniform1i(location, v0);
#endif
  } 
}


////////////////////////////////////////////////////////////////////////////////
void
program::set_uniform2i(char const* varname, GLint v0, GLint v1) const
{
  GLint location = get_uniform_location(varname);

  if (location >= 0) 
  {
#if GPUCAST_GL_DIRECT_STATE_ACCESS
    glProgramUniform2iEXT(id_, location, v0, v1);
#else 
    glUniform2i(location, v0, v1);
#endif
  } 
}


////////////////////////////////////////////////////////////////////////////////
void
program::set_uniform3i(char const* varname, GLint v0, GLint v1, GLint v2) const
{
  GLint location = get_uniform_location(varname);

  if (location >= 0) 
  {
#if GPUCAST_GL_DIRECT_STATE_ACCESS
    glProgramUniform3iEXT(id_, location, v0, v1, v2);
#else 
    glUniform3i(location, v0, v1, v2);
#endif
  } 
}


////////////////////////////////////////////////////////////////////////////////
void
program::set_uniform4i(char const* varname, GLint v0, GLint v1, GLint v2, GLint v3) const
{
  GLint location = get_uniform_location(varname);

  if (location >= 0) 
  {
#if GPUCAST_GL_DIRECT_STATE_ACCESS
    glProgramUniform4iEXT(id_, location, v0, v1, v2, v3);
#else 
    glUniform4i(location, v0, v1, v2, v3);
#endif
  } 
}


////////////////////////////////////////////////////////////////////////////////
void
program::set_uniform1fv(char const* varname, GLsizei count, GLfloat* value) const
{
  GLint location = get_uniform_location(varname);

  if (location >= 0) 
  {
#if GPUCAST_GL_DIRECT_STATE_ACCESS
    glProgramUniform1fvEXT(id_, location, count, value);
#else 
    glUniform1fv(location, count, value);
#endif
  } 
}


////////////////////////////////////////////////////////////////////////////////
void
program::set_uniform2fv(char const* varname, GLsizei count, GLfloat* value) const
{
  GLint location = get_uniform_location(varname);

  if (location >= 0) 
  {
#if GPUCAST_GL_DIRECT_STATE_ACCESS
    glProgramUniform2fvEXT(id_, location, count, value);
#else 
    glUniform2fv(location, count, value);
#endif
  } 
}


////////////////////////////////////////////////////////////////////////////////
void
program::set_uniform3fv(char const* varname, GLsizei count, GLfloat* value) const
{
  GLint location = get_uniform_location(varname);

  if (location >= 0) 
  {
#if GPUCAST_GL_DIRECT_STATE_ACCESS
    glProgramUniform3fvEXT(id_, location, count, value);
#else 
    glUniform3fv(location, count, value);
#endif
  }
}


////////////////////////////////////////////////////////////////////////////////
void
program::set_uniform4fv(char const* varname, GLsizei count, GLfloat* value) const
{
  GLint location = get_uniform_location(varname);

  if (location >= 0) 
  {
#if GPUCAST_GL_DIRECT_STATE_ACCESS
    glProgramUniform4fvEXT(id_, location, count, value);
#else 
    glUniform4fv(location, count, value);
#endif
  }
}


////////////////////////////////////////////////////////////////////////////////
void
program::set_uniform1iv(char const* varname, GLsizei count, GLint* value) const
{
  GLint location = get_uniform_location(varname);

  if (location >= 0) 
  {
#if GPUCAST_GL_DIRECT_STATE_ACCESS
    glProgramUniform1ivEXT(id_, location, count, value);
#else 
    glUniform1iv(location, count, value);
#endif
  }
}


////////////////////////////////////////////////////////////////////////////////
void
program::set_uniform2iv(char const* varname, GLsizei count, GLint* value) const
{
  GLint location = get_uniform_location(varname);

  if (location >= 0) 
  {
#if GPUCAST_GL_DIRECT_STATE_ACCESS
    glProgramUniform2ivEXT(id_, location, count, value);
#else 
    glUniform2iv(location, count, value);
#endif
  }
}


////////////////////////////////////////////////////////////////////////////////
void
program::set_uniform3iv(char const* varname, GLsizei count, GLint* value) const
{
  GLint location = get_uniform_location(varname);

  if (location >= 0) 
  {
#if GPUCAST_GL_DIRECT_STATE_ACCESS
    glProgramUniform3ivEXT(id_, location, count, value);
#else 
    glUniform3iv(location, count, value);
#endif
  }
}


////////////////////////////////////////////////////////////////////////////////
void
program::set_uniform4iv(char const* varname, GLsizei count, GLint* value) const
{
  GLint location = get_uniform_location(varname);

  if (location >= 0) 
  {
#if GPUCAST_GL_DIRECT_STATE_ACCESS
    glProgramUniform4ivEXT(id_, location, count, value);
#else 
    glUniform4iv(location, count, value);
#endif
  }
}


////////////////////////////////////////////////////////////////////////////////
void
program::set_uniform_matrix2fv(char const* varname, GLsizei count, GLboolean transpose, GLfloat* value) const
{
  GLint location = get_uniform_location(varname);

  if (location >= 0) 
  {
#if GPUCAST_GL_DIRECT_STATE_ACCESS
    glProgramUniformMatrix2fvEXT(id_, location, count, transpose, value);
#else 
    glUniformMatrix2fv(location, count, transpose, value);
#endif
  }
}


////////////////////////////////////////////////////////////////////////////////
void
program::set_uniform_matrix3fv(char const* varname, GLsizei count, GLboolean transpose, GLfloat* value) const
{
  GLint location = get_uniform_location(varname);

  if (location >= 0) 
  {
#if GPUCAST_GL_DIRECT_STATE_ACCESS
    glProgramUniformMatrix3fvEXT(id_, location, count, transpose, value);
#else 
    glUniformMatrix3fv(location, count, transpose, value);
#endif
  }
}


////////////////////////////////////////////////////////////////////////////////
void
program::set_uniform_matrix4fv(char const* varname, GLsizei count, GLboolean transpose, GLfloat const* value) const
{
  GLint location = get_uniform_location(varname);

  if (location >= 0) 
  {
#if GPUCAST_GL_DIRECT_STATE_ACCESS
    glProgramUniformMatrix4fvEXT(id_, location, count, transpose, value);
#else 
    glUniformMatrix4fv(location, count, transpose, value);
#endif
  }
}


////////////////////////////////////////////////////////////////////////////////
void        
program::set_uniform_matrix2x3fv( char const* varname, GLsizei count, GLboolean transpose, GLfloat* value ) const
{
  GLint location = get_uniform_location(varname);

  if (location >= 0) 
  {
#if GPUCAST_GL_DIRECT_STATE_ACCESS
    glProgramUniformMatrix2x3fvEXT(id_, location, count, transpose, value);
#else 
    glUniformMatrix2x3fv(location, count, transpose, value);
#endif
  }
}


////////////////////////////////////////////////////////////////////////////////
void        
program::set_uniform_matrix2x4fv( char const* varname, GLsizei count, GLboolean transpose, GLfloat* value ) const
{
  GLint location = get_uniform_location(varname);

  if (location >= 0) 
  {
#if GPUCAST_GL_DIRECT_STATE_ACCESS
    glProgramUniformMatrix2x4fvEXT(id_, location, count, transpose, value);
#else 
    glUniformMatrix2x4fv(location, count, transpose, value);
#endif
  }
}


////////////////////////////////////////////////////////////////////////////////
void        
program::set_uniform_matrix3x2fv( char const* varname, GLsizei count, GLboolean transpose, GLfloat* value ) const
{
  GLint location = get_uniform_location(varname);

  if (location >= 0) 
  {
#if GPUCAST_GL_DIRECT_STATE_ACCESS
    glProgramUniformMatrix3x2fvEXT(id_, location, count, transpose, value);
#else 
    glUniformMatrix3x2fv(location, count, transpose, value);
#endif
  }
}


////////////////////////////////////////////////////////////////////////////////
void        
program::set_uniform_matrix3x4fv( char const* varname, GLsizei count, GLboolean transpose, GLfloat* value ) const
{
  GLint location = get_uniform_location(varname);

  if (location >= 0) 
  {
#if GPUCAST_GL_DIRECT_STATE_ACCESS
    glProgramUniformMatrix3x4fvEXT(id_, location, count, transpose, value);
#else 
    glUniformMatrix3x4fv(location, count, transpose, value);
#endif
  }
}


////////////////////////////////////////////////////////////////////////////////
void        
program::set_uniform_matrix4x2fv( char const* varname, GLsizei count, GLboolean transpose, GLfloat* value ) const
{
  GLint location = get_uniform_location(varname);

  if (location >= 0) 
  {
#if GPUCAST_GL_DIRECT_STATE_ACCESS
    glProgramUniformMatrix4x2fvEXT(id_, location, count, transpose, value);
#else 
    glUniformMatrix4x2fv(location, count, transpose, value);
#endif
  }
}


////////////////////////////////////////////////////////////////////////////////
void        
program::set_uniform_matrix4x3fv( char const* varname, GLsizei count, GLboolean transpose, GLfloat* value ) const
{
  GLint location = get_uniform_location(varname);

  if (location >= 0) 
  {
#if GPUCAST_GL_DIRECT_STATE_ACCESS
    glProgramUniformMatrix4x3fvEXT(id_, location, count, transpose, value);
#else 
    glUniformMatrix4x3fv(location, count, transpose, value);
#endif
  }
}


////////////////////////////////////////////////////////////////////////////////
GLint
program::set_vertex_attrib(char const* varname) const
{
  assert(is_linked_);

  GLint index = glGetAttribLocation(id_, varname);
  glBindAttribLocation(id_, index, varname);

  return index;
}


////////////////////////////////////////////////////////////////////////////////
void 
program::set_texturebuffer(char const* varname, texturebuffer& t, GLint unit) const
{
  t.bind(unit);
  set_uniform1i(varname, unit);
}

////////////////////////////////////////////////////////////////////////////////
void        
program::set_shaderstoragebuffer(char const* varname, shaderstoragebuffer& ssbo, unsigned binding_point) const
{
  GLuint location = glGetProgramResourceIndex(id_, GL_SHADER_STORAGE_BLOCK, varname);

  ssbo.bind();
  glShaderStorageBlockBinding(id_, location, binding_point);
  ssbo.unbind();
}


////////////////////////////////////////////////////////////////////////////////
void 
program::set_texture1d(char const* varname, texture1d& t, GLint unit) const
{
  t.bind(unit);
  set_uniform1i(varname, unit);
}



////////////////////////////////////////////////////////////////////////////////
void 
program::set_texture2d(char const* varname, texture2d& t, GLint unit) const
{
  t.bind(unit);
  set_uniform1i(varname, unit);
}


////////////////////////////////////////////////////////////////////////////////
void 
program::set_texture3d(char const* varname, texture3d& t, GLint unit) const
{
  t.bind(unit);
  set_uniform1i(varname, unit);
}


////////////////////////////////////////////////////////////////////////////////
void 
program::set_texturearray(char const* varname, texturearray& t, GLint unit) const
{
  t.bind(unit);
  set_uniform1i(varname, unit);
}


////////////////////////////////////////////////////////////////////////////////
void 
program::set_cubemap(char const* varname, cubemap& c, GLint unit) const
{
  c.bind(unit);
  set_uniform1i(varname, unit);
}


////////////////////////////////////////////////////////////////////////////////
void 
program::bind_attrib_location(char const* varname, GLuint index) const
{
  assert(!is_linked_ && "bind_attrib_location() : trying to set location of an already linked program");
  glBindAttribLocation(id_, index, varname);
}


////////////////////////////////////////////////////////////////////////////////
void 
program::bind_fragdata_location(char const* varname, GLuint index) const
{
  assert(!is_linked_ && "bind_fragdata_location() : trying to set location of an already linked program");
  glBindFragDataLocation(id_, index, varname);
}


////////////////////////////////////////////////////////////////////////////////
GLint               
program::get_attrib_location(char const* varname) const
{
  return glGetAttribLocation(id_, varname);
}


////////////////////////////////////////////////////////////////////////////////
GLint
program::get_uniform_location(char const* varname) const
{
  return glGetUniformLocation(id_, varname);
}


////////////////////////////////////////////////////////////////////////////////
GLint               
program::get_uniform_blockindex(char const* varname) const
{
  return glGetUniformBlockIndex(id_, varname);
}


////////////////////////////////////////////////////////////////////////////////
GLint               
program::get_uniform_blocksize(char const* varname) const
{
  GLint blocksize;
  glGetActiveUniformBlockiv(id_, glGetUniformBlockIndex(id_, varname), GL_UNIFORM_BLOCK_DATA_SIZE, &blocksize);
  return blocksize;
}


////////////////////////////////////////////////////////////////////////////////
void                 
program::set_uniform_blockbinding(GLuint blockIndex, GLuint blockBinding ) const
{
  glUniformBlockBinding(id_, blockIndex, blockBinding);
}


////////////////////////////////////////////////////////////////////////////////
void                 
program::get_active_uniform_blockiv(GLuint uniformBlockIndex, GLenum pname, GLint *params) const
{
  glGetActiveUniformBlockiv(id_, uniformBlockIndex, pname, params);
}



} } // namespace gpucast / namespace gl
