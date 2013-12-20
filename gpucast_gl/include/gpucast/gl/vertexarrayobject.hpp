/********************************************************************************
* 
* Copyright (C) 2009 Bauhaus-Universitaet Weimar                                               
*
*********************************************************************************
*
*  module     : vertexarrayobject.hpp                                        
*  project    : glpp 
*  description: 
*
********************************************************************************/
#ifndef GPUCAST_GL_VERTEX_ARRAY_OBJECT_HPP
#define GPUCAST_GL_VERTEX_ARRAY_OBJECT_HPP

// header system
#include <string>

// header project
#include <gpucast/gl/glpp.hpp>




namespace gpucast { namespace gl {

  class buffer;

class GPUCAST_GL vertexarrayobject
{
public :
  
  vertexarrayobject                  ( );
  ~vertexarrayobject                 ( );

public : // methods

  void            bind            ( ) const;
  void            unbind          ( ) const;

  void            enable_attrib   ( std::size_t index ) const;
  void            disable_attrib  ( std::size_t index ) const;
  
  void            attrib_array    ( buffer const&         buf, 
                                    std::size_t           index, 
                                    std::size_t           size, 
                                    GLenum                type, 
                                    bool                  normalize,
                                    std::size_t           stride, 
                                    std::size_t           offset ) const;

private : // members

  GLuint          _id;
};

} } // namespace gpucast / namespace gl

#endif // GPUCAST_GL_ARRAY_BUFFER_HPP
