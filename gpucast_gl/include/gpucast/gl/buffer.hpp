/********************************************************************************
*
* Copyright (C) 2009-2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : buffer.hpp
*  project    : glpp
*  description:
*
********************************************************************************/
#ifndef GPUCAST_GL_BUFFER_HPP
#define GPUCAST_GL_BUFFER_HPP

#if WIN32
  #pragma warning(disable: 4996) // unsafe std::copy
#endif

// header system
#include <string>
#include <stdexcept>
#include <memory>

// header project
#include <gpucast/gl/glpp.hpp>


namespace gpucast { namespace gl {

class GPUCAST_GL buffer 
{
public : // c'tor, d'tor

  buffer                  ( GLenum usage = GL_STATIC_DRAW );
  buffer                  ( std::size_t bytes, GLenum usage = GL_STATIC_DRAW );

  virtual ~buffer         ( );

private : 

  buffer(buffer const& other) = delete;
  buffer& operator=(buffer const& other) = delete;

public : // methods

  void            swap            ( buffer& );

  void            bufferdata      ( std::size_t   size,
                                    GLvoid const* data,
                                    GLenum        usage = GL_STATIC_DRAW );

  void            buffersubdata   ( std::size_t   offset,
                                    std::size_t   size,
                                    GLvoid const* data ) const;

  void            getbuffersubdata ( GLintptr     offset,
                                     GLsizeiptr   size,
                                     GLvoid*      data ) const;

  template <typename iterator_type>
  void            update          ( iterator_type begin, iterator_type end);

  GLuint64EXT     address         ( ) const;

  GLuint          id              ( ) const;
  GLenum          usage           ( ) const;
  std::size_t     capacity        ( ) const;
  GLint           parameter       ( GLenum parameter ) const;

  void*           map             ( GLenum access ) const;

  void*           map_range       ( GLintptr    offset,  
                                    GLsizeiptr  length,  
                                    GLbitfield  access ) const;

  bool            unmap           ( ) const;

  void            make_resident   ( ) const;

  void            bind_base       (unsigned binding_point) const;
  void            bind_range      (unsigned binding_point, std::size_t in_offset, std::size_t in_size);
  void            unbind          (unsigned binding_point);

  void            clear_data      (GLenum internal_format, GLenum format, GLenum type, void* data);
  void            clear_subdata   (GLenum internal_format, unsigned offset, unsigned size, GLenum format, GLenum type, void* data);

  virtual void    bind            ( ) const = 0;
  virtual void    unbind          ( ) const = 0;
  virtual GLenum  target          ( ) const = 0;

private : // members

  GLuint          _id;
  GLenum          _usage;
  std::size_t     _capacity;
};

typedef std::shared_ptr<buffer>         buffer_ptr;

  template <typename iterator_type>
  void buffer::update ( iterator_type begin, iterator_type end)
  {
    typedef typename std::iterator_traits<iterator_type>::value_type value_type;
    std::size_t const growth   = 1.2;

    std::size_t elemsize = sizeof(value_type);
    std::size_t elements = std::distance(begin, end);

    if ( elements == 0 ) return;

    if ( elemsize * elements > _capacity)  // reallocate new memory 
    {
      _capacity = std::size_t(elemsize * elements * growth);
      bufferdata(_capacity, 0);
    }

    // map buffer, write data and unmap buffer
    //void* target = map_range(0, _capacity, GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_RANGE_BIT );
#if 0
    void* target = map ( GL_WRITE_ONLY );

    if ( target != 0 ) {
      std::copy(begin, end, static_cast<value_type*>(target));
    } else {
      throw std::runtime_error("buffer::update(): out of memory.");
    }

    unmap();
#else
    buffersubdata ( 0, elemsize * elements, (GLvoid*)(&(*begin)));
#endif
  }

} } // namespace gpucast / namespace gl

#endif // GPUCAST_GL_BUFFER_HPP
