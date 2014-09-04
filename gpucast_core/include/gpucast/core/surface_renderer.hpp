/********************************************************************************
*
* Copyright (C) 2007-2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : surface_renderer.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_CORE_RENDERSERVICE_HPP
#define GPUCAST_CORE_RENDERSERVICE_HPP

// header, system
#include <string>
#include <set>

// header, external
#include <gpucast/math/matrix4x4.hpp>

// header, project
#include <gpucast/core/gpucast.hpp>
#include <gpucast/core/renderer.hpp>



namespace gpucast {

class beziersurfaceobject;

class GPUCAST_CORE surface_renderer : public renderer
{
public : // enums, typedefs

  enum trimapproach_type { double_binary_partition = 0,
                           contourmap_binary       = 1,
                           contourmap_kdtree       = 2 };

  typedef beziersurfaceobject             drawable_type;
  typedef std::shared_ptr<drawable_type>  drawable_ptr;

public : // c'tor / d'tor

  surface_renderer           ( int argc, char** argv );
  virtual ~surface_renderer  ();

public : // methods

  void                  newton_iterations   ( std::size_t );
  std::size_t           newton_iterations   ( ) const;

  void                  cube_mapping        ( bool );
  bool                  cube_mapping        ( ) const;

  void                  sphere_mapping      ( bool );
  bool                  sphere_mapping      ( ) const;

  void                  diffuse_mapping     ( bool );
  bool                  diffuse_mapping     ( ) const;

  void                  trimming            ( bool );
  bool                  trimming            ( ) const;

  void                  trim_approach       ( trimapproach_type a );
  trimapproach_type     trim_approach       ( ) const;

  void                  raycasting          ( bool );
  bool                  raycasting          ( ) const;

  virtual drawable_ptr  create              () = 0;

  virtual void          draw                () = 0;

  //virtual void          draw                ( drawable_ptr const& drawable ) = 0;

  virtual void          spheremap           ( std::string const& filepath ) = 0;

  virtual void          diffusemap          ( std::string const& filepath ) = 0;

  virtual void          cubemap             ( std::string const& positive_x,
                                              std::string const& negative_x,
                                              std::string const& positive_y,
                                              std::string const& negative_y,
                                              std::string const& positive_z,
                                              std::string const& negative_z ) = 0;

protected : // attributes 

  virtual void          _init_shader        () = 0;

protected : // attributes                     

  bool                    _cube_mapping;
  bool                    _sphere_mapping;
  bool                    _diffuse_mapping;
  bool                    _trimming_enabled;
  bool                    _raycasting_enabled;

  trimapproach_type       _trimapproach;

  std::size_t             _newton_iterations;
};

} // namespace gpucast

#endif // GPUCAST_CORE_RENDERSERVICE_HPP
