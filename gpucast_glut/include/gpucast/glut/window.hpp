/********************************************************************************
*
* Copyright (C) 2009 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : window.hpp
*  project    : glpp
*  description:
*
********************************************************************************/
#ifndef GPUCAST_GLUT_WINDOW_HPP
#define GPUCAST_GLUT_WINDOW_HPP

// header, system
#include <map>
#include <set>
#include <memory>
#include <functional>

// header, project
#include <gpucast/glut/gpucast_glut.hpp>
#include <gpucast/gl/util/eventhandler.hpp>

namespace gpucast { namespace gl {

class GPUCAST_GLUT glutwindow 
{
  public : // typedef

    typedef std::function<void (int,int)>   keyfunction_t;
    typedef std::function<void ()>          drawfunction_t;
    typedef std::shared_ptr<eventhandler>   eventhandler_ptr;

  private : // singleton

    glutwindow                    ( int         argc,
                                    char*       argv[],
                                    std::size_t width,
                                    std::size_t height,
                                    std::size_t posx,
                                    std::size_t posy,
                                    std::size_t context_major = 3,
                                    std::size_t context_minor = 3,
                                    bool        core_context = false);

    glutwindow(glutwindow const& other) = delete;
    glutwindow& operator=(glutwindow const& other) = delete;

  public :

    virtual ~glutwindow    ( );

    static glutwindow& instance();

    static void   init          ( int         argc,
                                  char*       argv[],
                                  std::size_t width,
                                  std::size_t height,
                                  std::size_t posx,
                                  std::size_t posy,
                                  std::size_t context_major = 3,
                                  std::size_t context_minor = 3,
                                  bool        core_context = false);

    void         run            ();
    void         leave          ( std::function<void()> f ) const;

    std::size_t  width          ( ) const;
    std::size_t  height         ( ) const;

    std::size_t  posx           ( ) const;
    std::size_t  posy           ( ) const;

    void         set_drawfunction(std::shared_ptr <std::function<void()>> const& f);

    void         add_keyevent   ( unsigned char  key,
                                  keyfunction_t  event );

    void         add_eventhandler    ( eventhandler_ptr );
    void         remove_eventhandler ( eventhandler_ptr );

    void         printinfo      ( std::ostream& ) const;


  public :

    static void display         ( );
    static void reshape         ( int w, int h);
    static void idle            ( );
    static void keyboard        ( unsigned char key, int x, int y );
    static void mouse           ( int button, int state, int x, int y );
    static void motion          ( int x, int y );
    static void passivemotion   ( int x, int y );

  private : // methods

    int         _init           ( int argc, char* argv[] );
    // keyboard event handling
    void        _keysetup       ( );
    void        _fullscreen     ( int, int );
    void        _quit           ( int, int );

  protected : // members

    std::size_t                         _width;
    std::size_t                         _height;
    std::size_t                         _posx;
    std::size_t                         _posy;

    std::map<char, keyfunction_t>       _keymap;
    std::set<eventhandler_ptr>          _eventhandler;
    std::shared_ptr<std::function<void()>>  _drawfun;

    int                                 _handle;
    std::size_t                         _contextmajor;
    std::size_t                         _contextminor;
    bool                                _core;
};

} } // namespace gpucast / namespace gl

#endif // GPUCAST_GLUT_WINDOW_HPP

