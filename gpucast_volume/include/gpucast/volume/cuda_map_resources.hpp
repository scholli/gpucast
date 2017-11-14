/********************************************************************************
*
* Copyright (C) 2009-2011 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : cuda_gl_image.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_CUDA_GL_RESOURCES_HPP
#define GPUCAST_CUDA_GL_RESOURCES_HPP

#include <iostream>
#include <exception>

//#include <GL/glew.h>

#include <gpucast/gl/buffer.hpp>

#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>

/////////////////////////////////////////////////////////////////////////////
inline void register_image ( struct cudaGraphicsResource **resource, GLuint image, GLenum target, unsigned int flags )
{
  if ( !(*resource) ) {
    std::cout << "Register GL Resources for Image ..." << std::endl;
    cudaError_t cuda_err = cudaGraphicsGLRegisterImage( resource, image, target, flags );
    if ( cuda_err != cudaSuccess ) {
      std::cerr << " cudaGraphicsGLRegisterImage failed ... " << std::endl;
    }
  } else {
    std::cerr << " CudaRessource already registered ... " << std::endl;
  }
}

/////////////////////////////////////////////////////////////////////////////
inline void register_buffer( struct cudaGraphicsResource **resource, gpucast::gl::buffer const& buffer, unsigned int flags )
{
  if ( !(*resource) && buffer.capacity() != 0 ) {
    std::cout << "Register GL Resources for Buffer ..." << std::endl;
    cudaError_t cuda_err = cudaGraphicsGLRegisterBuffer( resource, buffer.id(), flags );
    if ( cuda_err != cudaSuccess ) {
      std::cerr << " cudaGraphicsGLRegisterImage failed ... " << std::endl;
    }
  } else {
    if (*resource) std::cerr << " Cannot register_buffer. CudaRessource already registered." << std::endl;
    if (buffer.capacity() == 0) std::cerr << " Cannot register_buffer. Buffer empty." << std::endl;
  }
}

///////////////////////////////////////////////////////////////////////////////
inline void unregister_resource ( struct cudaGraphicsResource** resource )
{
  if ( *resource ) 
  {
    cudaError_t err = cudaGraphicsUnregisterResource ( *resource );
    if ( err == cudaSuccess ) 
    {
      *resource = 0;
    } else {
      throw std::runtime_error ( "unregister CUDA resource failed" );
    }
  }
}


///////////////////////////////////////////////////////////////////////////////
inline void 
map_resource ( struct cudaGraphicsResource*  resource )
{
  cudaError_t err;

  // map gl resource and retrieve device_ptr
  err = cudaGraphicsMapResources(1, &resource, 0);
  if ( err != cudaSuccess )
  {
    std::cout << "Mapping GL Resource failed. " << cudaGetErrorString(err) << std::endl;
  }
}


///////////////////////////////////////////////////////////////////////////////
inline void 
map_resources ( unsigned n, struct cudaGraphicsResource**  resources )
{
  cudaError_t err = cudaSuccess;

  // map gl resource and retrieve device_ptr
  err = cudaGraphicsMapResources(n, resources, 0);
  if ( err != cudaSuccess )
  {
    std::cout << "Mapping GL Resources failed. " << cudaGetErrorString(err) << std::endl;
  }
}



///////////////////////////////////////////////////////////////////////////////
inline void 
unmap_resource ( struct cudaGraphicsResource*  resource )
{
  cudaError_t err;

  // map gl resource and release device_ptr
  err = cudaGraphicsUnmapResources(1, &resource, 0);
  if ( err != cudaSuccess )
  {
    std::cout << "Unmapping GL Resource failed. " << cudaGetErrorString(err) << std::endl;
  }
}


///////////////////////////////////////////////////////////////////////////////
inline void 
unmap_resources ( unsigned n, struct cudaGraphicsResource**  resources )
{
  cudaError_t err = cudaSuccess;

  // map gl resource and release device_ptr
  err = cudaGraphicsUnmapResources(n, resources, 0);
  if ( err != cudaSuccess )
  {
    std::cout << "Unmapping GL Resources failed. " << cudaGetErrorString(err) << std::endl;
  }
}



///////////////////////////////////////////////////////////////////////////////
template <typename surface_t>
inline void 
bind_mapped_resource_to_surface ( struct cudaGraphicsResource* resource, surface_t* image )
{
  cudaError_t err;
  cudaArray*  resource_array;

  err = cudaGraphicsSubResourceGetMappedArray(&resource_array, resource, 0, 0);
  if ( err != cudaSuccess )
  {
    std::cout << "Get Device Pointer failed. " << cudaGetErrorString(err) << std::endl;
  }

  cudaChannelFormatDesc desc;
  err = cudaGetChannelDesc(&desc, resource_array);
  if ( err != cudaSuccess )
  {
    std::cout << "Get Image Format failed. " << cudaGetErrorString(err) << std::endl;
  }

  std::cout << image << std::endl;

  //err = cudaBindSurfaceToArray(image, resource_array, &desc);
  if ( err != cudaSuccess )
  {
    switch (err)
    {
      case cudaErrorInvalidValue : 
        std::cout << "Bind Surface to Array failed. cudaErrorInvalidValue. " << cudaGetErrorString(err) << std::endl;
      break;
      case cudaErrorInvalidSurface :
        std::cout << "Bind Surface to Array failed. cudaErrorInvalidSurface. " << cudaGetErrorString(err) << std::endl;
        break;
    };
    
  }
}


///////////////////////////////////////////////////////////////////////////////
template <typename texture_t>
inline void 
bind_mapped_resource_to_texture ( struct cudaGraphicsResource* resource, texture_t* tex )
{
  cudaError_t err;
  cudaArray*  resource_array;

  err = cudaGraphicsSubResourceGetMappedArray(&resource_array, resource, 0, 0);
  if ( err != cudaSuccess )
  {
    std::cout << "Get Device Pointer failed. " << cudaGetErrorString(err) << std::endl;
  }

  cudaChannelFormatDesc desc;
  err = cudaGetChannelDesc(&desc, resource_array);
  if ( err != cudaSuccess )
  {
    std::cout << "Get Image Format failed. " << cudaGetErrorString(err) << std::endl;
  }

  err = cudaBindTextureToArray(tex, resource_array, &desc);
  if ( err != cudaSuccess )
  {
    std::cout << "Bind Texture to Array failed. " << cudaGetErrorString(err) << std::endl;
  }
}

///////////////////////////////////////////////////////////////////////////////
template <typename pointer_t>
inline void 
bind_mapped_resource_to_pointer ( struct cudaGraphicsResource* resource, pointer_t& p )
{
  std::size_t bytes;
  cudaError_t err = cudaGraphicsResourceGetMappedPointer ( (void**)&p, &bytes, resource ); 
  if ( err != cudaSuccess )
  {
    std::cout << "Bind Resource to Pointer failed. " << cudaGetErrorString(err) << std::endl;
  }
}

#endif