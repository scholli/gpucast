/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : renderer.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_CORE_RENDERER_HPP
#define GPUCAST_CORE_RENDERER_HPP

// header, system
#include <string>
#include <set>
#include <memory>

#include <gpucast/math/vec4.hpp>
#include <gpucast/math/matrix4x4.hpp>

// header, external

// header, project
#include <gpucast/core/gpucast.hpp>


namespace gpucast {

class GPUCAST_CORE renderer 
{
public : // enums

  renderer(renderer const& other) = delete;
  renderer& operator=(renderer const& other) = delete;

public : // c'tor / d'tor

  renderer           ();
  virtual ~renderer  ();

public : // methods

  void                  add_path                ( std::string const& path, std::string separator = ":" );

  float                 nearplane               () const;
  float                 farplane                () const;

  void                  nearplane               ( float );
  void                  farplane                ( float );

  virtual void          modelviewmatrix         ( gpucast::math::matrix4f const& );
  virtual void          projectionmatrix        ( gpucast::math::matrix4f const& );

  gpucast::math::matrix4f const& modelviewmatrix         () const;
  gpucast::math::matrix4f const& modelviewmatrixinverse  () const;
  gpucast::math::matrix4f const& modelviewprojection     () const;
  gpucast::math::matrix4f const& modelviewprojectioninverse () const;
  gpucast::math::matrix4f const& normalmatrix            () const;
  gpucast::math::matrix4f const& projectionmatrix        () const;

  void                  background              ( gpucast::math::vec4f const& rgba );
  gpucast::math::vec4f const&    background              ( ) const;

  virtual void          resize                  ( int width, int height );

protected : // attributes 

  std::pair<bool, std::string>          _path_to_file   ( std::string const& filename ) const;

protected : // attributes                     

  int                                   _width;
  int                                   _height;

  bool                                  _initialized_cuda;

  std::set<std::string>                 _pathlist;
  gpucast::math::vec4f                    _background;

  float                                 _nearplane;
  float                                 _farplane;

  gpucast::math::matrix4f                 _modelviewmatrix;
  gpucast::math::matrix4f                 _modelviewmatrixinverse;
  gpucast::math::matrix4f                 _projectionmatrix;
  gpucast::math::matrix4f                 _normalmatrix;
  gpucast::math::matrix4f                 _modelviewprojectionmatrix;
  gpucast::math::matrix4f                 _modelviewprojectionmatrixinverse;
};

} // namespace gpucast

#endif // GPUCAST_CORE_RENDERSERVICE_HPP
