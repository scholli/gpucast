/********************************************************************************
*
* Copyright (C) 2015 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : resource_factory.cpp
*  project    : gpucast/gl
*  description:
*
********************************************************************************/
#include <gpucast/gl/util/resource_factory.hpp>

#include <fstream>
#include <sstream>
#include <iostream>
#include <locale>

#if WIN32
  #include <codecvt>
#endif

#include <boost/regex.hpp>
#include <boost/filesystem.hpp>
#include <boost/log/trivial.hpp>

#include <gpucast/core/config.hpp>

#include <gpucast/gl/program.hpp>
#include <gpucast/gl/shader.hpp>

namespace {

  std::string gen_line(std::size_t line, std::string const& label) {
    return std::string("#line " + std::to_string(line) + " \"" + label + "\"\n");
  };

  std::wstring gen_line(std::size_t line, std::wstring const& label) {
    std::wstring label_prepared = label;
    std::replace(label_prepared.begin(), label_prepared.end(), '\\', '/');
    return std::wstring(L"#line " + std::to_wstring(line) + L" \"" + label_prepared + L"\"\n");
  };

};

namespace gpucast {
  namespace gl {

////////////////////////////////////////////////////////////////////////////////

resource_factory::resource_factory(std::vector<std::string> const& shader_root_directories)
    : _search_paths(shader_root_directories)
{
  add_search_path(std::string(GPUCAST_INSTALL_DIR));
  add_search_path(std::string(GPUCAST_INSTALL_DIR) + "/resources");
  add_search_path(std::string(GPUCAST_INSTALL_DIR) + "/resources/shaders");
}

////////////////////////////////////////////////////////////////////////////////

void resource_factory::add_search_path(std::string const& path)
{
  _search_paths.push_back(path);
}

////////////////////////////////////////////////////////////////////////////////

std::string resource_factory::read_plain_file(std::string const& path) const
{
  namespace fs = boost::filesystem;
  std::string out;
  fs::path p;
  if (!get_file_contents(fs::path(path), fs::current_path(), out, p)) {
    throw std::runtime_error("Unable to read plain file");
  }

  return out;
}

////////////////////////////////////////////////////////////////////////////////

std::string resource_factory::read_shader_file(std::string const& path) const
{
  namespace fs = boost::filesystem;

  std::string out;
  if (!resolve_includes(fs::path(path), fs::current_path(), out)) {
    throw std::runtime_error("Unable to read shader from file");
  }
  return std::string(GPUCAST_GLSL_VERSION_STRING) + out;
}

////////////////////////////////////////////////////////////////////////////////

std::string resource_factory::prepare_shader(std::string const& shader_source,
                                            std::string const& label) const
{
  namespace fs = boost::filesystem;

  std::string out = shader_source;
  if (!resolve_includes(fs::path(), fs::current_path(), out, label)) {
    throw std::runtime_error("Unable to prepare shader");
  }

  return std::string(GPUCAST_GLSL_VERSION_STRING) + out;
}

////////////////////////////////////////////////////////////////////////////////

std::string resource_factory::resolve_substitutions(std::string const& shader_source,
  substition_map const& smap) const
{
  //TODO: add support for the #line macro if multi-line substitutions are supplied.
  boost::regex regex("\\@(\\w+)\\@");
  boost::smatch match;
  std::string out, s = shader_source;

  while (boost::regex_search(s, match, regex)) {
    std::string subs;
    auto search = smap.find(match[1]);
    if (search != smap.end()) {
      subs = search->second;
    }
    else {
      BOOST_LOG_TRIVIAL(error) << "Option \"" << match[1] << "\" is unknown!" << std::endl;
      subs = match.str();
    }
    out += match.prefix().str() + subs;
    s = match.suffix().str();
  }
  return out + s;
}

////////////////////////////////////////////////////////////////////////////////
std::shared_ptr<program> resource_factory::create_program(std::vector<shader_desc> const& shader_descs) const {
  try {
    auto new_program = std::make_shared<program>();

    for (auto desc : shader_descs) {
      gpucast::gl::shader shader(desc.type);
      auto shader_source = read_shader_file(desc.filename);

      shader.set_source(shader_source.c_str());
      if (!shader.log().empty()) {
        BOOST_LOG_TRIVIAL(info) << desc.filename << " log : " << shader.log() << std::endl;
      }
      new_program->add(&shader);
    }
    
    // link all shaders
    new_program->link();

    if (!new_program->log().empty()) {
      BOOST_LOG_TRIVIAL(info) << " program log : " << new_program->log() << std::endl;
    }

    return new_program;
  }
  catch (std::exception& e) {
    BOOST_LOG_TRIVIAL(error) << "resource_factory::create_program(): failed to init program : " << e.what() << "\n";
    return nullptr;
  }
}


////////////////////////////////////////////////////////////////////////////////

bool resource_factory::get_file_contents(boost::filesystem::path const& filename,
                                        boost::filesystem::path const& current_dir,
                                        std::string& contents,
                                        boost::filesystem::path& full_path) const
{
  std::ifstream ifs;
  std::stringstream error_info;

  auto probe = [&](boost::filesystem::path const& dir) -> bool {
    full_path = boost::filesystem::absolute(filename, dir);
    ifs.open(full_path.native());
    if (!ifs) {
      //std::string directory_native = wstring_converter.to_bytes();
      error_info << "[" << filename.string() << "]: " << strerror(errno) << std::endl;
      return false;
    } else {
      return true;
    }
  };

  // probe files
  if (!probe(current_dir)) {
    for (auto const& path : _search_paths) {
      if (probe(boost::filesystem::path(path))) break;
    }
  }

  // log error if failed to find file
  if (!ifs) {
    BOOST_LOG_TRIVIAL(error) << "Failed to get file: \"" << filename.string()
                             << "\" from any of the search paths:" << std::endl
                             << error_info.str() << std::endl;
    contents = "";
    full_path = boost::filesystem::path();
    return false;
  }

  contents.assign((std::istreambuf_iterator<char>(ifs)),
                  (std::istreambuf_iterator<char>()));
  full_path = boost::filesystem::canonical(full_path);
  return true;
}

////////////////////////////////////////////////////////////////////////////////

bool resource_factory::get_file_contents(boost::filesystem::path const& filename,
  boost::filesystem::path const& current_dir,
  std::wstring& contents,
  boost::filesystem::path& full_path) const
{
  std::ifstream ifs;
  std::stringstream error_info;

  auto probe = [&](boost::filesystem::path const& dir) -> bool {
    full_path = boost::filesystem::absolute(filename, dir);
    ifs.open(full_path.native());
    if (!ifs) {
      error_info << "[" << filename.string() << "]: " << strerror(errno) << std::endl;
      return false;
    }
    else {
      return true;
    }
  };

  // probe files
  if (!probe(current_dir)) {
    for (auto const& path : _search_paths) {
      if (probe(boost::filesystem::path(path))) break;
    }
  }

  // log error if failed to find file
  if (!ifs) {
    BOOST_LOG_TRIVIAL(error) << "Failed to get file: \"" << filename.string()
      << "\" from any of the search paths:" << std::endl
      << error_info.str() << std::endl;
    contents = L"";
    full_path = boost::filesystem::path();
    return false;
  }

  contents.assign((std::istreambuf_iterator<char>(ifs)),
                  (std::istreambuf_iterator<char>()));
  full_path = boost::filesystem::canonical(full_path);
  return true;
}


////////////////////////////////////////////////////////////////////////////////

bool resource_factory::resolve_includes(boost::filesystem::path const& filename,
                                       boost::filesystem::path const& current_dir,
                                       std::string& contents,
                                       std::string const& custom_label) const
{
  // get contents
  typedef boost::filesystem::path::string_type string_type; 
  typedef string_type::value_type char_type;

  string_type s;
  string_type file_label;

  boost::filesystem::path    first_search_dir;
  if (filename.empty()) {
    // take shader code from 'contents' parameter and label from 'custom_label'
    s = string_type(contents.begin(), contents.end());
    file_label = string_type(custom_label.begin(), custom_label.end());
    first_search_dir = current_dir;
  }
  else {
    // load shader code from file
    boost::filesystem::path full_path;
    if (!get_file_contents(filename, current_dir, s, full_path)) {
      contents = "";
      return false;
    }

    file_label = full_path.native();
    first_search_dir = full_path.parent_path();
  }

  // substitute inclusions
  s = gen_line(1, file_label) + s;

  std::string regex_as_char_t = "(\\@|\\#)\\s*include\\s*\"([^\"]+)\"";
  std::string newline_as_char_t = "\n";

  boost::basic_regex<char_type> regex(regex_as_char_t.begin(), regex_as_char_t.end());
  string_type newline(newline_as_char_t.begin(), newline_as_char_t.end());

  boost::match_results<string_type::const_iterator> match;

  string_type out;
  std::size_t line_ctr{};

  while (boost::regex_search(s, match, regex)) 
  {
    std::string shader_code;

    if (!resolve_includes(match[2].str(), first_search_dir, shader_code)) {
      contents = "";
      return false;
    }
    string_type prefix = match.prefix().str();
    line_ctr += std::count(prefix.begin(), prefix.end(), '\n');

    string_type shader_code_native(shader_code.begin(), shader_code.end());
    out += prefix + newline + shader_code_native + newline + gen_line(line_ctr, file_label);
    s = match.suffix().str();
  }

#if WIN32
  typedef std::codecvt_utf8<wchar_t> convert_type;
  
  std::wstring_convert<convert_type, wchar_t> converter;
  auto contents_native = out + s;

  contents = converter.to_bytes(contents_native);
#else
  contents = out + s;
#endif

  return true;
}

} } // namespace gpucast / namespace gl

