/********************************************************************************
*
* Copyright (C) 2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : dynamic_test.cpp
*  project    : glpp
*  description:
*
********************************************************************************/
// include i/f header
#include "gpucast/gl/test/dynamic_test.hpp"

#include <gpucast/gl/glpp.hpp>

namespace gpucast { namespace gl {

  ///////////////////////////////////////////////////////////////////////////////
  dynamic_test::dynamic_test()
    : _actions(),
      _action_index(0),
      _timer(),
      _total_time(),
      _frame_times(),
      _frames(0),
      _initialized(false)
  {}


  ///////////////////////////////////////////////////////////////////////////////
  dynamic_test::~dynamic_test()
  {}


  ///////////////////////////////////////////////////////////////////////////////
  void                  
  dynamic_test::init ()
  {
    std::for_each ( _initfunctions.begin(), _initfunctions.end(), [] ( std::function<void()> const& f ) { f(); } );
    _initialized = true;
  }


  ///////////////////////////////////////////////////////////////////////////////
  bool 
  dynamic_test::initialized () const
  {
    return _initialized;
  }


  ///////////////////////////////////////////////////////////////////////////////
  void                  
  dynamic_test::add_init_function ( std::function<void()> const& f )
  {
    _initfunctions.push_back(f);
  }


  ///////////////////////////////////////////////////////////////////////////////
  void                  
  dynamic_test::begin_frame ()
  {
    _timer.start();
  }


  ///////////////////////////////////////////////////////////////////////////////
  void                  
  dynamic_test::end_frame ()
  {
    glFinish();

    ++_frames;

    _timer.stop();

    time_duration frametime = _timer.result();
    _frame_times.push_back(frametime);
    _total_time = _total_time + frametime;
  }


  ///////////////////////////////////////////////////////////////////////////////
  unsigned                  
  dynamic_test::frames ()
  {
    return _frames;
  }


  ///////////////////////////////////////////////////////////////////////////////
  gpucast::math::matrix4f        
  dynamic_test::get_current_modeltransform () const
  {
    if ( _action_index > _actions.size() - 1 )
    {
      return gpucast::math::matrix4f();
    } else {
      if ( current_transform()->finished() )
      {
        _action_index;
      }
      return current_transform()->current_transform();
    }
  }


  ///////////////////////////////////////////////////////////////////////////////
  time_duration const&
  dynamic_test::get_total_time () const
  {
    return _total_time;
  }

  ///////////////////////////////////////////////////////////////////////////////
  float 
  dynamic_test::get_average_fps() const
  {
    return float(_frames)/float(_total_time.as_seconds());
  }

  ///////////////////////////////////////////////////////////////////////////////
  bool                  
  dynamic_test::finished () const
  {
    assert ( _action_index <= _actions.size() - 1 );

    if ( _actions.empty() ) 
    {
      return true;
    } else {
      if ( _action_index == _actions.size() - 1 )
      {
        return current_transform()->finished();
      } else {
        return false;
      }
    }
  }


  ///////////////////////////////////////////////////////////////////////////////
  void                 
  dynamic_test::reset ()
  {
    std::for_each(_actions.begin(), _actions.end(), std::bind(&dynamic_transform::reset, std::placeholders::_1));

    _frame_times.clear();

    _total_time   = time_duration();
    _action_index = 0;
    _frames       = 0;
    _initialized  = false;
  }


  ///////////////////////////////////////////////////////////////////////////////
  void                  
  dynamic_test::step ()
  {
    if ( !finished() )
    {
      if ( current_transform()->finished() )
      {
        ++_action_index;
      }

      current_transform()->step();
    }
  }


  ///////////////////////////////////////////////////////////////////////////////
  dynamic_transform_ptr 
  dynamic_test::current_transform () const
  {
    std::list<dynamic_transform_ptr>::const_iterator action = _actions.begin();
    std::advance(action, _action_index);
    return *action;
  }


  ///////////////////////////////////////////////////////////////////////////////
  std::list<time_duration> const& 
  dynamic_test::frametimes  () const
  {
    return _frame_times;
  }


  ///////////////////////////////////////////////////////////////////////////////
  void                   
  dynamic_test::add ( dynamic_transform_ptr const& action )
  {
    _actions.push_back ( action );
  }


  ///////////////////////////////////////////////////////////////////////////////
  void                  
  dynamic_test::remove ( dynamic_transform_ptr const& action )
  {
    _actions.remove(action);
  }


} } // namespace gpucast / namespace gl

