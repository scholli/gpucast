/********************************************************************************
* 
* Copyright (C) 2010 Bauhaus-Universitaet Weimar                                               
*
*********************************************************************************
*
*  module     : cubemap.hpp                                        
*  project    : glpp 
*  description: 
*
********************************************************************************/
#ifndef GPUCAST_GL_CUBEMAP_HPP
#define GPUCAST_GL_CUBEMAP_HPP

// header system
#include <string>

// header project
#include <gpucast/gl/glpp.hpp>
#include <gpucast/gl/texture.hpp>

#include <boost/noncopyable.hpp>


namespace gpucast { namespace gl {

  class GPUCAST_GL cubemap : public texture, public boost::noncopyable
  {
  public :
    
    cubemap                 ( );
    ~cubemap                ( );
    void          swap      ( cubemap& );

  public : // methods

    void          load      ( std::string const& imgfile_posx,
                              std::string const& imgfile_negx,
                              std::string const& imgfile_posy,
                              std::string const& imgfile_negy,
                              std::string const& imgfile_posz,
                              std::string const& imgfile_negz );

    GLuint const  id        ( ) const override;

    void          bind      ( );

    void          bind      ( GLint texunit );

    void          unbind    ( );

    void          enable    ( ) const;

    void          disable   ( ) const;
    
    void          parameter ( GLenum pname, int param );

    static GLenum target    ( );

  private : // methods

    void          openfile_ ( std::string const&  filename, 
                              GLenum              target );

  private : // members

    GLuint        id_;
    GLint         unit_;
  };

} } // namespace gpucast / namespace gl

#endif // GPUCAST_GL_CUBEMAP_HPP
