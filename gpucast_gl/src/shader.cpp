/********************************************************************************
*
* Copyright (C) 2009 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : shader.cpp
*  project    : glpp
*  description:
*
********************************************************************************/

// header i/f
#include "gpucast/gl/shader.hpp"


#include <boost/config/warning_disable.hpp>

#define BOOST_FILESYSTEM_NO_DEPRECATED
#include <boost/filesystem/fstream.hpp>
#include <boost/filesystem/operations.hpp>

// header, system
#include <iostream>
#include <string>
#include <cstring>
#include <cstdlib>
#include <istream>
#include <sstream>

#include <GL/glew.h>

// header, project

namespace gpucast { namespace gl {

///////////////////////////////////////////////////////////////////////////////
shader::shader()
{}

///////////////////////////////////////////////////////////////////////////////
shader::~shader()
{
  glDeleteShader(id_);
}

///////////////////////////////////////////////////////////////////////////////
void
shader::load(std::string const& fileName)
{
  std::string file = file_content(fileName);

  replace_includes(file);

  std::size_t srclen = file.length();

  char* src = new char[srclen+1];
  strcpy(src, file.c_str());

  set_source(src);

  delete[] src;
}

///////////////////////////////////////////////////////////////////////////////
bool
shader::compile() const
{
  GLint compiled, type;

  glCompileShader(id_);

  glGetShaderiv(id_, GL_COMPILE_STATUS, &compiled);
  glGetShaderiv(id_, GL_SHADER_TYPE, &type);

  std::string logfilename;

  switch (type) {
    case GL_VERTEX_SHADER :
      logfilename.append("vertexshader"); break;
    case GL_FRAGMENT_SHADER :
      logfilename.append("fragmentshader"); break;
    case GL_TESS_CONTROL_SHADER :
      logfilename.append("tessellation_controlshader"); break;
    case GL_TESS_EVALUATION_SHADER :
      logfilename.append("tessellation_evaluationshader"); break;
    case GL_GEOMETRY_SHADER :
      logfilename.append("geometryshader"); break;
    default :
      logfilename.append("unknown_shadertype");
  };

  if (!compiled)
  {
    logfilename.append(".fail.log");
    std::fstream fstr(logfilename.c_str(), std::ios::out);
    fstr << get_source() << std::endl << std::endl << log() << std::endl;
    return false;
  } else {
    logfilename.append(".success.log");
    std::fstream fstr(logfilename.c_str(), std::ios::out);
    fstr << get_source() << std::endl << std::endl << log() << std::endl;
    return true;
  }
}

///////////////////////////////////////////////////////////////////////////////
void
shader::set_source(char const* src)
{
  glShaderSource(id_, 1, (const char **)&src, 0);
}

///////////////////////////////////////////////////////////////////////////////
std::string
shader::get_source() const
{
  GLint id_len;
  glGetShaderiv(id_, GL_SHADER_SOURCE_LENGTH, &id_len);

  GLint size;
  char* buf = new char[id_len];
  glGetShaderSource(id_, id_len, &size, buf);

  std::string source(buf);
  delete[] buf;

  return source;
}

///////////////////////////////////////////////////////////////////////////////
std::string
shader::log ( ) const
{
  GLint log_len;
  glGetShaderiv(id_, GL_INFO_LOG_LENGTH, &log_len);

  GLint size;
  char* buf = new char[log_len];

  glGetShaderInfoLog(id_, log_len, &size, buf);

  std::string infolog(buf);
  delete[] buf;

  return infolog;
}


///////////////////////////////////////////////////////////////////////////////
GLuint
shader::id()
{
  return (id_);
}


////////////////////////////////////////////////////////////////////////////////
std::string
shader::file_content(std::string const& filename )
{
  int length(0);
  char* buffer;
  std::ifstream input(filename.c_str());

  if (!input.is_open()) {
    std::cerr << "Warning: Can't open file "<< filename << std::endl;
    return std::string();
  }

  // Get length of file:
  input.seekg(0, std::ios::end);
  length = int(input.tellg());
  input.seekg(0, std::ios::beg);

  buffer = new char[length];
  input.getline(buffer, length, '\0');
  input.close();

  std::string str(buffer);
  delete[] buffer;

  return str;
}

////////////////////////////////////////////////////////////////////////////////
void
shader::replace_includes(std::string& buffer)
{
	std::size_t b, e;

	while (find_include(buffer, b, e)) 
  {
		std::string dirline(buffer, b, e-b);
		std::size_t fst = std::min(dirline.find('"'), dirline.find('<'));
    ++fst;
		std::size_t lst = std::min(dirline.find('"', fst), dirline.find('>', fst));

		if (fst < dirline.size() && lst < dirline.size())
    {
			std::string filename(dirline, fst, lst-fst);
      std::string srccode = file_content(filename.c_str());
			buffer.erase(b, e-b);
			buffer.insert(b, srccode);
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
bool
shader::find_include(std::string const& buffer, std::size_t& b, std::size_t& e)
{
	b = buffer.find("#include");
	if (b < buffer.size()) {
		e = buffer.find('\n', b);
	}
	return b < buffer.size();
}


} } // namespace gpucast / namespace gl
