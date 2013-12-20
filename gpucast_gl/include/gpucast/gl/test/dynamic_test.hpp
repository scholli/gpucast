/********************************************************************************
*
* Copyright (C) 2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : dynamic_test.hpp
*  project    : glpp
*  description:
*
********************************************************************************/
#ifndef GPUCAST_GL_DYNAMIC_TEST_HPP
#define GPUCAST_GL_DYNAMIC_TEST_HPP

#include <boost/shared_ptr.hpp>
#include <boost/function.hpp>

#include <list>

#include <gpucast/gl/glpp.hpp>
#include <gpucast/gl/util/timer.hpp>
#include <gpucast/gl/math/matrix4x4.hpp>
#include <gpucast/gl/math/vec4.hpp>
#include <gpucast/gl/test/dynamic_transform.hpp>

namespace gpucast { namespace gl {

  class GPUCAST_GL dynamic_test
  {
  public : // typedefs

  public : // c'tor / d'tor

    dynamic_test();
    ~dynamic_test();

  public : // methods

    void                  init                        ();
    bool                  initialized                 () const;
    void                  add_init_function           ( std::function<void()> const& );

    void                  begin_frame                 ();
    void                  end_frame                   ();
    unsigned              frames                      ();

    matrix4f              get_current_modeltransform  () const;
    time_duration const&  get_total_time              () const;
    float                 get_average_fps             () const;

    bool                  finished                    () const;
    void                  reset                       ();
    void                  step                        ();

    dynamic_transform_ptr current_transform           () const;
    std::list<time_duration> const& frametimes  () const;

    void                  add                         ( dynamic_transform_ptr const& );
    void                  remove                      ( dynamic_transform_ptr const& );

  private : // attributes

    std::list<dynamic_transform_ptr>    _actions;
    unsigned                            _action_index;

    timer                         _timer;
    time_duration                 _total_time;
    std::list<time_duration>      _frame_times;

    unsigned                            _frames;
    std::list<std::function<void()>>  _initfunctions;
    bool                                _initialized;

  };

} } // namespace gpucast / namespace gl

#endif // GPUCAST_GL_DYNAMIC_TEST_HPP
