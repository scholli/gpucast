/********************************************************************************
*
* Copyright (C) 2009 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : filereader.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/

#ifndef FILEREADER_HPP
#define FILEREADER_HPP

// header, system
#include <map>
#include <list>
#include <sstream>

// header, project
#include <gpucast/gl/util/material.hpp>
#include <gpucast/gl/math/vec3.hpp>
#include <gpucast/core/surface_renderer.hpp>

// forward declarations
namespace gpucast {
  class beziersurfaceobject;
}

class filereader
{
  public :
    typedef std::map<std::size_t, std::size_t> map_t;

  public :


    void read_igs(gpucast::surface_renderer& renderer,
                  std::string const& s,
                  gpucast::gl::material const& mat,
                  std::vector<std::shared_ptr<gpucast::beziersurfaceobject> >& bl) const;

    void read_conf(gpucast::surface_renderer& renderer,
                   std::string const& s,
                   std::vector<std::shared_ptr<gpucast::beziersurfaceobject> >& bl,
                   gpucast::gl::vec3f& background_color) const;

  private :

    void parse_material_conf(std::istringstream& sstr,
                             gpucast::gl::material& mat) const;

    bool parse_float(std::istringstream& sstr,
                     float& result) const;

    void parse_background(std::istringstream& sstr,
                          gpucast::gl::vec3f&) const;
};

#endif // FILEREADER_HPP
