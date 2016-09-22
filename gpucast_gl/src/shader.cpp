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
#include <boost/log/trivial.hpp>

// header, system
#include <iostream>
#include <string>
#include <cstring>
#include <cstdlib>
#include <istream>
#include <sstream>

#include <GL/glew.h>

#include <gpucast/gl/error.hpp>

// header, project

namespace gpucast { namespace gl {

///////////////////////////////////////////////////////////////////////////////
shader::shader(shader_type type)
  : _id(0),
    _type(type)
{
  _create(_type);
}

///////////////////////////////////////////////////////////////////////////////
shader::shader(shader_type type, std::string const& filename)
  : _id(0),
    _type(type)
{
  _create(type);
  load(filename);
}

///////////////////////////////////////////////////////////////////////////////
shader::shader(shader_desc const& desc)
  : _id(0),
    _type(desc.type)
{
  _create(_type);
  load(desc.filename);
}

///////////////////////////////////////////////////////////////////////////////
shader::~shader()
{
  reset();
}

///////////////////////////////////////////////////////////////////////////////
void shader::reset() {
  glDeleteShader(_id);
}

///////////////////////////////////////////////////////////////////////////////
void
shader::load(std::string const& filename)
{
  std::string file_content = _file_content(filename);

  _replace_includes(file_content);

  set_source(file_content.c_str());
}

///////////////////////////////////////////////////////////////////////////////
void shader::load(shader_type type, std::string const& filename)
{
  _create(type);
  load(filename);
}


///////////////////////////////////////////////////////////////////////////////
bool
shader::compile() const
{
  glCompileShader(_id);

  GLint compiled = 0;
  glGetShaderiv(_id, GL_COMPILE_STATUS, &compiled);

  std::string logfilename;

  switch (_type) {
    case vertex_stage :
      logfilename.append("vertexshader"); break;
    case tesselation_control_stage:
      logfilename.append("tessellation_controlshader"); break;
    case tesselation_evaluation_stage:
      logfilename.append("tessellation_evaluationshader"); break;
    case geometry_stage:
      logfilename.append("geometryshader"); break;
    case fragment_stage:
      logfilename.append("fragmentshader"); break;
    default :
      logfilename.append("unknown_shadertype");
  };

  if (!compiled)
  {
    BOOST_LOG_TRIVIAL(info) << "Compiling " << logfilename << " failed." << std::endl;
    logfilename.append(".fail.log");
    std::fstream fstr(logfilename.c_str(), std::ios::out);
    fstr << get_source() << std::endl << std::endl << log() << std::endl;
    fstr.close();
    return false;
  } else {
    BOOST_LOG_TRIVIAL(info) << "Compiling " << logfilename << " succeed." << std::endl;
    logfilename.append(".success.log");
    std::fstream fstr(logfilename.c_str(), std::ios::out);
    fstr << get_source() << std::endl << std::endl << log() << std::endl;
    fstr.close();
    return true;
  }
}

///////////////////////////////////////////////////////////////////////////////
void
shader::set_source(char const* src)
{
  glShaderSource(_id, 1, (const char **)&src, 0);
  compile();
}

///////////////////////////////////////////////////////////////////////////////
std::string
shader::get_source() const
{
  GLint _idlen;
  glGetShaderiv(_id, GL_SHADER_SOURCE_LENGTH, &_idlen);

  GLint size;
  char* buf = new char[_idlen];
  glGetShaderSource(_id, _idlen, &size, buf);

  std::string source(buf);
  delete[] buf;

  return source;
}

///////////////////////////////////////////////////////////////////////////////
std::string
shader::log () const
{
  GLint log_len;
  glGetShaderiv(_id, GL_INFO_LOG_LENGTH, &log_len);

  if (log_len != 0) {
    GLint size;
    std::string info_log;
    info_log.resize(log_len);
    
    glGetShaderInfoLog(_id, log_len, &size, &info_log[0]);

    return info_log;
  }
  else {
    return std::string();
  }
}


///////////////////////////////////////////////////////////////////////////////
GLuint
shader::id() const
{
  return (_id);
}

////////////////////////////////////////////////////////////////////////////////
shader_type shader::type() const
{
  return _type;
}

////////////////////////////////////////////////////////////////////////////////
void shader::_create(shader_type type) 
{
  // delete old shaer if available
  reset();

  // store new type 
  _type = type;
  
  // create GL object
  switch (_type) {
  case vertex_stage:
    _id = glCreateShader(GL_VERTEX_SHADER);
    break;
  case tesselation_control_stage:
    _id = glCreateShader(GL_TESS_CONTROL_SHADER);
    break;
  case tesselation_evaluation_stage:
    _id = glCreateShader(GL_TESS_EVALUATION_SHADER);
    break;
  case geometry_stage:
    _id = glCreateShader(GL_GEOMETRY_SHADER);
    break;
  case fragment_stage:
    _id = glCreateShader(GL_FRAGMENT_SHADER);
    break;
  }
}


////////////////////////////////////////////////////////////////////////////////
std::string
shader::_file_content(std::string const& filename )
{
  int length(0);
  char* buffer;
  std::ifstream input(filename.c_str());

  if (!input.is_open()) {
    BOOST_LOG_TRIVIAL(error) << "Warning: Can't open file "<< filename << std::endl;
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
shader::_replace_includes(std::string& buffer)
{
	std::size_t b, e;

	while (_find_include(buffer, b, e)) 
  {
		std::string dirline(buffer, b, e-b);
		std::size_t fst = std::min(dirline.find('"'), dirline.find('<'));
    ++fst;
		std::size_t lst = std::min(dirline.find('"', fst), dirline.find('>', fst));

		if (fst < dirline.size() && lst < dirline.size())
    {
			std::string filename(dirline, fst, lst-fst);
      std::string srccode = _file_content(filename.c_str());
			buffer.erase(b, e-b);
			buffer.insert(b, srccode);
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
bool
shader::_find_include(std::string const& buffer, std::size_t& b, std::size_t& e)
{
	b = buffer.find("#include");
	if (b < buffer.size()) {
		e = buffer.find('\n', b);
	}
	return b < buffer.size();
}


} } // namespace gpucast / namespace gl
