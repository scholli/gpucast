/********************************************************************************
*
* Copyright (C) 2009 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : camera.hpp
*  project    : glpp
*  description:
*
********************************************************************************/
#ifndef GPUCAST_GL_CAMERA_HPP
#define GPUCAST_GL_CAMERA_HPP

#include <functional>

#include <gpucast/gl/glpp.hpp>
#include <gpucast/gl/math/vec3.hpp>

namespace gpucast {
  namespace gl {

    class GPUCAST_GL camera
    {
    public: // c'tor / d'tor

      camera();
      virtual ~camera();

    public: // methods

      virtual void  drawcallback(std::function<void()>);
      virtual void  resize(std::size_t resolution_x, std::size_t resolution_y);
      virtual void  draw();

      void          target(vec3f const&);
      vec3f const&  target() const;

      void          position(vec3f const&);
      vec3f const&  position() const;

      void          up(vec3f const&);
      vec3f const&  up() const;

      void          nearplane(float);
      float         nearplane() const;

      void          farplane(float);
      float         farplane() const;

      void          screenoffset_y(float);
      float         screenoffset_y() const;

      void          screenoffset_x(float);
      float         screenoffset_x() const;

      void          screenwidth(float);
      float         screenwidth() const;

      void          screenheight(float);
      float         screenheight() const;

      void          screendistance(float);
      float         screendistance() const;

      std::size_t   resolution_x() const;
      std::size_t   resolution_y() const;

    protected: // member

      std::function<void()> _drawcallback;

      vec3f                   _target;
      vec3f                   _position;
      vec3f                   _up;

      float                   _znear;
      float                   _zfar;

      float                   _screenoffset_y;
      float                   _screenoffset_x;

      float                   _screenwidth;
      float                   _screenheight;

      float                   _screendistance;

      std::size_t             _resolution_x;
      std::size_t             _resolution_y;
    };

} } // namespace gpucast / namespace gl

#endif // GPUCAST_GL_CAMERA_HPP
