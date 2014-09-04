/********************************************************************************
* 
* Copyright (C) 2009 Bauhaus-Universitaet Weimar                                               
*
*********************************************************************************
*
*  module     : material.hpp                                        
*  project    : glpp 
*  description: 
*
********************************************************************************/
#ifndef GPUCAST_GL_MATERIAL_HPP
#define GPUCAST_GL_MATERIAL_HPP

#include <memory>
#include <cstdlib>
#include <algorithm>

#include <gpucast/gl/glpp.hpp>
#include <gpucast/math/vec4.hpp>
#include <gpucast/gl/texture2d.hpp>

namespace gpucast { namespace gl {

  struct GPUCAST_GL material 
  {
    material()
      : ambient   (0.0f, 0.0f, 0.0f, 1.0f),
        diffuse   (0.0f, 0.0f, 0.0f, 1.0f),
        specular  (0.0f, 0.0f, 0.0f, 1.0f),
        shininess (20.0f),
        opacity   (1.0f)
    {}

    virtual ~material()
    {}

    material(
             gpucast::math::vec4f const& amb,
             gpucast::math::vec4f const& diff,
	           gpucast::math::vec4f const& spec,
	           float shine,
	           float opac
             )
      : ambient     (amb),
        diffuse     (diff),
        specular    (spec),
        shininess   (shine),
        opacity     (opac),
        ambientmap  (),
        diffusemap  (),
        specularmap (),
        bumpmap     (),
        normalmap   ()
    {}

    void randomize ( float max_ambient = 1.0f, float max_diffuse = 1.0f, float max_specular = 1.0f, float max_shininess = 1.0f, float min_opacity = 1.0f )
    {
      std::for_each(&ambient[0],  (&ambient[3]) + 1,  [&] ( float& x ) { x = max_ambient * float(std::rand()) / RAND_MAX; } );
      std::for_each(&diffuse[0],  (&diffuse[3]) + 1,  [&] ( float& x ) { x = max_diffuse * float(std::rand()) / RAND_MAX; } );
      std::for_each(&specular[0], (&specular[3]) + 1, [&] ( float& x ) { x = max_specular * float(std::rand()) / RAND_MAX; } );
      shininess = max_shininess * float(std::rand()) / RAND_MAX;
      opacity   = std::max(min_opacity, float(std::rand()) / RAND_MAX);
    }

    void print (std::ostream& os) const
    {
      os << "Ka = " << ambient << std::endl;
      os << "Kd = " << diffuse << std::endl;
      os << "Ks = " << specular << std::endl;

      os << "opacity   = " << opacity << std::endl;
      os << "shininess = " << shininess << std::endl;

      os << "has ambient map : "  << int(ambientmap != 0)   << std::endl;
      os << "has diffuse map : "  << int(diffusemap != 0)   << std::endl;
      os << "has specular map : " << int(specularmap != 0)  << std::endl;
      os << "has bump map : "     << int(bumpmap != 0)      << std::endl;
      os << "has normal map : "   << int(normalmap != 0)    << std::endl;
    }

    gpucast::math::vec4f        ambient;
    gpucast::math::vec4f        diffuse;
    gpucast::math::vec4f        specular;

    float                       shininess;
    float                       opacity;

    std::shared_ptr<texture2d>  ambientmap;
    std::shared_ptr<texture2d>  diffusemap;
    std::shared_ptr<texture2d>  specularmap;
  
    std::shared_ptr<texture2d>  bumpmap;
    std::shared_ptr<texture2d>  normalmap;
    std::shared_ptr<texture2d>  opacitymap;
  };

} } // namespace gpucast / namespace gl

#endif // GPUCAST_GL_MATERIAL_HPP
