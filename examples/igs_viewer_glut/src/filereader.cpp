/********************************************************************************
*
* Copyright (C) 2009 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : filereader.cpp
*  project    : gpucast
*  description:
*
********************************************************************************/
// i/f header
#include "filereader.hpp"

// header, system
#include <fstream>

// header, project
#include <gpucast/core/beziersurfaceobject.hpp>
#include <gpucast/core/surface_converter.hpp>
#include <gpucast/core/nurbssurfaceobject.hpp>
#include <gpucast/core/import/igs_loader.hpp>
#include <gpucast/core/surface_renderer.hpp>

#include <gpucast/gl/math/vec3.hpp>


using namespace gpucast;

///////////////////////////////////////////////////////////////////////////////
void
filereader::read_igs(gpucast::surface_renderer& renderer,
                     std::string const& s,
                     gpucast::gl::material const& mat,
                     std::vector<std::shared_ptr<gpucast::beziersurfaceobject> >& bl) const
{
  std::shared_ptr<gpucast::nurbssurfaceobject>   n_obj(new gpucast::nurbssurfaceobject);
  std::shared_ptr<gpucast::beziersurfaceobject>  b_obj = renderer.create();
  bl.push_back(b_obj);

  std::cout << "Loading : " << s << std::endl;

  if (s.find(".igs") == s.length() - 4)
  {
    gpucast::igs_loader loader;
    loader.load(s, n_obj);
  } else {
    std::cerr << "application::addFile(): unknown file extension. exiting." << std::endl;
    exit(0);
  }

  std::cout << "NURBS surfaces : " << n_obj->surfaces() << std::endl;
  std::cout << "NURBS trimming curves : " << n_obj->trimcurves() << std::endl;

  gpucast::surface_converter conv;
  std::cout << "Converting to Bezier..." << std::endl;
  conv.convert(n_obj, b_obj);
  std::cout << "BEZIER surfaces : " << b_obj->surfaces() << std::endl;
  std::cout << "BEZIER trimming curves : " << b_obj->trimcurves() << std::endl;

  // apply current material
  b_obj->material(mat);
}

///////////////////////////////////////////////////////////////////////////////
void
filereader::read_conf(gpucast::surface_renderer& renderer,
                      std::string const& s,
                      std::vector<std::shared_ptr<gpucast::beziersurfaceobject> >& bl,
                      gpucast::gl::vec3f& background_color) const
{
  std::ifstream ifstr(s.c_str());
  typedef std::vector<std::pair<gpucast::gl::material, std::string> > file_map_t;
  file_map_t filemap;
  gpucast::gl::material current_material;

  if (ifstr.good()) {
    std::string line;
    while (ifstr) {
      std::getline(ifstr, line);

      if (!line.empty()) {
        std::istringstream sstr(line);
        std::string qualifier;
        sstr >> qualifier;

        // if not comment line
        if (qualifier.size() > 0) {
          if (qualifier.at(0) != '#') {
            // define material
            if (qualifier == "material") {
              parse_material_conf(sstr, current_material);
            }
            // load igs file
            if (qualifier == "object") {
              if (sstr) {
                std::string filename;
                sstr >> filename;
                filemap.push_back(std::make_pair(current_material, filename));
              }
            }
            if (qualifier == "background") {
              if (sstr) {
                parse_background(sstr, background_color);
              }
            }
          }
        }
      }
    }
  } else {
    std::cerr << "application::open_configfile(): cannot find config file : " << s << std::endl;
  }

  for (file_map_t::iterator i = filemap.begin(); i != filemap.end(); ++i) {
    read_igs(renderer, i->second, i->first, bl);
  }

  ifstr.close();
}

///////////////////////////////////////////////////////////////////////////////
void
filereader::parse_material_conf(std::istringstream& sstr,
                                gpucast::gl::material& mat) const
{
  float ar, ag, ab, dr, dg , db, sr, sg, sb, shine, opac;

  // ambient coefficients
  parse_float(sstr, ar);
  parse_float(sstr, ag);
  parse_float(sstr, ab);

  // diffuse coefficients
  parse_float(sstr, dr);
  parse_float(sstr, dg);
  parse_float(sstr, db);

  // specular coefficients
  parse_float(sstr, sr);
  parse_float(sstr, sg);
  parse_float(sstr, sb);

  // shininess
  parse_float(sstr, shine);

  // opacity
  if (parse_float(sstr, opac)) {
    mat.ambient   = gpucast::gl::vec3f(ar, ag, ab);
    mat.diffuse   = gpucast::gl::vec3f(dr, dg, db);
    mat.specular  = gpucast::gl::vec3f(sr, sg, sb);
    mat.shininess = shine;
    mat.opacity   = opac;
	} else {
	  std::cerr << "application::read_material(): material definition incomplete. discarding.\n usage: material ar ab ag   dr dg db  sr sg sb  shininess   opacity";
	}
}


///////////////////////////////////////////////////////////////////////////////
bool
filereader::parse_float(std::istringstream& sstr,
                        float& result) const
{
  if (sstr) {
    sstr >> result;
    return true;
  } else {
    return false;
  }
}

///////////////////////////////////////////////////////////////////////////////
void
filereader::parse_background(std::istringstream& sstr,
                             gpucast::gl::vec3f& bg) const
{
  float r, g, b;
  sstr >> r >> g >> b;
  bg = gpucast::gl::vec3f(r, g, b);
}
