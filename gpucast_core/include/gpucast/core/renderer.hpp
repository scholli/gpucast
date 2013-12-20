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

#include <gpucast/gl/glpp.hpp>
#include <gpucast/gl/math/vec4.hpp>
#include <gpucast/gl/math/matrix4x4.hpp>
#include <gpucast/gl/program.hpp>

// header, external
#include <boost/noncopyable.hpp>

// header, project
#include <gpucast/core/gpucast.hpp>


namespace gpucast {

class GPUCAST_CORE renderer : public boost::noncopyable
{
public : // enums

public : // c'tor / d'tor

  renderer           ( int argc, char** argv );
  virtual ~renderer  ();

public : // methods

  void                  add_path                ( std::string const& path, std::string separator = ":" );

  float                 nearplane               () const;
  float                 farplane                () const;

  void                  nearplane               ( float );
  void                  farplane                ( float );

  virtual void          modelviewmatrix         ( gpucast::gl::matrix4f const& );
  virtual void          projectionmatrix        ( gpucast::gl::matrix4f const& );

  gpucast::gl::matrix4f const& modelviewmatrix         () const;
  gpucast::gl::matrix4f const& modelviewmatrixinverse  () const;
  gpucast::gl::matrix4f const& modelviewprojection     () const;
  gpucast::gl::matrix4f const& modelviewprojectioninverse () const;
  gpucast::gl::matrix4f const& normalmatrix            () const;
  gpucast::gl::matrix4f const& projectionmatrix        () const;

  void                  background              ( gpucast::gl::vec4f const& rgba );
  gpucast::gl::vec4f const&    background              ( ) const;

  virtual void          resize                  ( int width, int height );

  void                  init_program            ( std::shared_ptr<gpucast::gl::program>& program,
                                                  std::string const&                vertexshader_filename,
                                                  std::string const&                fragmentshader_filename,
                                                  std::string const&                geometryshader_filename = "" );

  int                   cuda_get_max_flops_device_id() const;

protected : // attributes 

  std::pair<bool, std::string>          _path_to_file   ( std::string const& filename ) const;

protected : // attributes                     

  int                                   _argc;
  char**                                _argv;

  int                                   _width;
  int                                   _height;

  bool                                  _initialized_cuda;

  std::set<std::string>                 _pathlist;
  gpucast::gl::vec4f                    _background;

  float                                 _nearplane;
  float                                 _farplane;

  gpucast::gl::matrix4f                 _modelviewmatrix;
  gpucast::gl::matrix4f                 _modelviewmatrixinverse;
  gpucast::gl::matrix4f                 _projectionmatrix;
  gpucast::gl::matrix4f                 _normalmatrix;
  gpucast::gl::matrix4f                 _modelviewprojectionmatrix;
  gpucast::gl::matrix4f                 _modelviewprojectionmatrixinverse;
};

} // namespace gpucast

#endif // GPUCAST_CORE_RENDERSERVICE_HPP
