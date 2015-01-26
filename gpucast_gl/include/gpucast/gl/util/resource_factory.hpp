/********************************************************************************
*
* Copyright (C) 2015 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : resource_factory.hpp
*  project    : gpucast/gl
*  description:
*
********************************************************************************/

#ifndef GPUCAST_GL_RESOURCE_FACTORY_HPP
#define GPUCAST_GL_RESOURCE_FACTORY_HPP

#include <vector>
#include <map>
#include <unordered_map>
#include <list>
#include <boost/filesystem.hpp>

#include <gpucast/gl/glpp.hpp>

namespace gpucast {
  namespace gl {

    class program;

    typedef std::unordered_map<std::string, std::string> substition_map;

    class GPUCAST_GL resource_factory {

    public:

      resource_factory(std::vector<std::string> const& search_directories = std::vector<std::string>());

      virtual ~resource_factory() {}

      void         add_search_path(std::string const& path);

      std::string  read_plain_file(std::string const& file) const;

      std::string  read_shader_file(std::string const& file) const;

      std::string  prepare_shader(std::string const& shader_source,
        std::string const& label) const;

      std::string  resolve_substitutions(std::string const& shader_source,
        substition_map const& smap) const;

      std::shared_ptr<program> create_program(std::string const& vertex_shader, std::string const& fragment_shader) const;

    private:

      bool         get_file_contents(boost::filesystem::path const& filename,
        boost::filesystem::path const& current_dir,
        std::string& contents,
        boost::filesystem::path& full_path) const;

      bool         get_file_contents(boost::filesystem::path const& filename,
        boost::filesystem::path const& current_dir,
        std::wstring& contents,
        boost::filesystem::path& full_path) const;

      bool         resolve_includes(boost::filesystem::path const& filename,
        boost::filesystem::path const& current_dir,
        std::string& contents,
        std::string const& custom_label = std::string()) const;

      std::vector<std::string> _search_paths;

    };

  }
} // namespace gpucast / namespace gl

#endif  // GPUCAST_GL_RESOURCE_FACTORY_HPP
