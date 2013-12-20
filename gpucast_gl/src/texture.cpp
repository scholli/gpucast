/********************************************************************************
*
* Copyright (C) 2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : texture.cpp
*  project    : glpp
*  description:
*
********************************************************************************/
#include "gpucast/gl/texture.hpp"

// header system

// disable boost warnings
#if WIN32
  #pragma warning (disable : 4511)
  #pragma warning (disable : 4127)
#endif

// header dependencies
#include <GL/glew.h>

// header project

namespace gpucast { namespace gl {

static std::map<GLenum, texture_format> _texture_formats;

///////////////////////////////////////////////////////////////////////////////
texture_format::texture_format()
  : internal_format(GL_NONE),
    base_format(GL_NONE),
    size(0),
    type(GL_NONE),
    name("GL_NONE")
{}

///////////////////////////////////////////////////////////////////////////////
texture_format::texture_format(GLenum internal_fmt, GLenum base_fmt, std::size_t components_size, GLenum value_type, std::string const& as_string)
  : internal_format(internal_fmt),
    base_format(base_fmt),
    size(components_size),
    type(value_type),
    name(as_string)
{}

///////////////////////////////////////////////////////////////////////////////
texture::texture( )
{
  _init();
}


///////////////////////////////////////////////////////////////////////////////
texture::~texture( )
{}


///////////////////////////////////////////////////////////////////////////////
std::size_t             
texture::size_of_format ( GLenum internal_format )
{
  _init();
  if ( _texture_formats.count(internal_format) ) 
  {
    return _texture_formats.at(internal_format).size;
  } else {
    return 0;
  }
}

///////////////////////////////////////////////////////////////////////////////
GLenum                  
texture::base_format ( GLenum internal_format )
{
  _init();
  if ( _texture_formats.count(internal_format) ) 
  {
    return _texture_formats.at(internal_format).base_format;
  } else {
    return 0;
  }
}


///////////////////////////////////////////////////////////////////////////////
GLenum                  
texture::value_type_of_format  ( GLenum internal_format )
{
  _init();
  if ( _texture_formats.count(internal_format) ) 
  {
    return _texture_formats.at(internal_format).type;
  } else {
    return 0;
  }
}


///////////////////////////////////////////////////////////////////////////////
std::string             
texture::name ( GLenum internal_format )
{
  _init();
  if ( _texture_formats.count(internal_format) ) 
  {
    return _texture_formats.at(internal_format).name;
  } else {
    return std::string();
  }
}


///////////////////////////////////////////////////////////////////////////////
void
texture::_init()
{
  if ( _texture_formats.empty() )
  {
    _texture_formats[GL_R8]                 = texture_format( GL_R8,                  GL_RED,              1, GL_UNSIGNED_BYTE,          "GL_R8" );
    _texture_formats[GL_R8_SNORM]           = texture_format( GL_R8_SNORM,            GL_RED,              1, GL_BYTE,                   "GL_R8_SNORM" );         
    _texture_formats[GL_R16]                = texture_format( GL_R16,                 GL_RED,              2, GL_UNSIGNED_SHORT,         "GL_R16" );              
    _texture_formats[GL_R16_SNORM]          = texture_format( GL_R16_SNORM,           GL_RED,              2, GL_SHORT,                  "GL_R16_SNORM" );        
    _texture_formats[GL_RG8]                = texture_format( GL_RG8,                 GL_RG,               2, GL_UNSIGNED_BYTE,          "GL_RG8" );              
    _texture_formats[GL_RG8_SNORM]          = texture_format( GL_RG8_SNORM,           GL_RG,               2, GL_BYTE,                   "GL_RG8_SNORM" );        
    _texture_formats[GL_RG16]               = texture_format( GL_RG16,                GL_RG,               4, GL_UNSIGNED_SHORT,         "GL_RG16" );             
    _texture_formats[GL_RG16_SNORM]         = texture_format( GL_RG16_SNORM,          GL_RG,               4, GL_SHORT,                  "GL_RG16_SNORM" );       
    _texture_formats[GL_R3_G3_B2]           = texture_format( GL_R3_G3_B2,            GL_RGB,              1, GL_UNSIGNED_BYTE_3_3_2,    "GL_R3_G3_B2" );         
    _texture_formats[GL_RGB4]               = texture_format( GL_RGB4,                GL_RGB,              2, GL_UNSIGNED_SHORT_4_4_4_4, "GL_RGB4" );             
    _texture_formats[GL_RGB5]               = texture_format( GL_RGB5,                GL_RGB,              2, GL_UNSIGNED_SHORT_5_5_5_1, "GL_RGB5" );             
    _texture_formats[GL_RGB8]               = texture_format( GL_RGB8,                GL_RGB,              3, GL_UNSIGNED_BYTE,          "GL_RGB8" );             
    _texture_formats[GL_RGB8_SNORM]         = texture_format( GL_RGB8_SNORM,          GL_RGB,              3, GL_BYTE,                   "GL_RGB8_SNORM" );       
    _texture_formats[GL_RGB10]              = texture_format( GL_RGB10,               GL_RGB,              4, GL_UNSIGNED_INT_10_10_10_2,"GL_RGB10" );            
    _texture_formats[GL_RGB12]              = texture_format( GL_RGB12,               GL_RGB,              4, GL_UNSIGNED_SHORT,         "GL_RGB12" );            
    _texture_formats[GL_RGB16]              = texture_format( GL_RGB16,               GL_RGB,              6, GL_UNSIGNED_SHORT,         "GL_RGB16" );            
    _texture_formats[GL_RGB16_SNORM]        = texture_format( GL_RGB16_SNORM,         GL_RGB,              6, GL_SHORT,                  "GL_RGB16_SNORM" );      
    _texture_formats[GL_RGBA2]              = texture_format( GL_RGBA2,               GL_RGBA,             1, GL_UNSIGNED_BYTE,          "GL_RGBA2" );            
    _texture_formats[GL_RGBA4]              = texture_format( GL_RGBA4,               GL_RGBA,             2, GL_UNSIGNED_SHORT_4_4_4_4, "GL_RGBA4" );            
    _texture_formats[GL_RGB5_A1]            = texture_format( GL_RGB5_A1,             GL_RGBA,             2, GL_UNSIGNED_SHORT_5_5_5_1, "GL_RGB5_A1" );          
    _texture_formats[GL_RGBA8]              = texture_format( GL_RGBA8,               GL_RGBA,             4, GL_UNSIGNED_SHORT,         "GL_RGBA8" );            
    _texture_formats[GL_RGBA8_SNORM]        = texture_format( GL_RGBA8_SNORM,         GL_RGBA,             4, GL_BYTE,                   "GL_RGBA8_SNORM" );      
    _texture_formats[GL_RGB10_A2]           = texture_format( GL_RGB10_A2,            GL_RGBA,             4, GL_UNSIGNED_INT_10_10_10_2,"GL_RGB10_A2" );         
    _texture_formats[GL_RGB10_A2UI]         = texture_format( GL_RGB10_A2UI,          GL_RGBA,             4, GL_UNSIGNED_INT_10_10_10_2,"GL_RGB10_A2UI" );       
    _texture_formats[GL_RGBA12]             = texture_format( GL_RGBA12,              GL_RGBA,             6, GL_UNSIGNED_SHORT,         "GL_RGBA12" );           
    _texture_formats[GL_RGBA16]             = texture_format( GL_RGBA16,              GL_RGBA,             8, GL_UNSIGNED_SHORT,         "GL_RGBA16" );           
    _texture_formats[GL_RGBA16_SNORM]       = texture_format( GL_RGBA16_SNORM,        GL_RGBA,             8, GL_SHORT,                  "GL_RGBA16_SNORM" );     
    _texture_formats[GL_SRGB8]              = texture_format( GL_SRGB8,               GL_RGB,              3, GL_BYTE,                   "GL_SRGB8" );            
    _texture_formats[GL_SRGB8_ALPHA8]       = texture_format( GL_SRGB8_ALPHA8,        GL_RGBA,             4, GL_BYTE,                   "GL_SRGB8_ALPHA8" );     
    _texture_formats[GL_R16F]               = texture_format( GL_R16F,                GL_RED,              2, GL_FLOAT,                  "GL_R16F" );             
    _texture_formats[GL_RG16F]              = texture_format( GL_RG16F,               GL_RG,               4, GL_FLOAT,                  "GL_RG16F" );            
    _texture_formats[GL_RGB16F]             = texture_format( GL_RGB16F,              GL_RGB,              6, GL_FLOAT,                  "GL_RGB16F" );           
    _texture_formats[GL_RGBA16F]            = texture_format( GL_RGBA16F,             GL_RGBA,             8, GL_FLOAT,                  "GL_RGBA16F" );          
    _texture_formats[GL_R32F]               = texture_format( GL_R32F,                GL_RED,              4, GL_FLOAT,                  "GL_R32F" );             
    _texture_formats[GL_RG32F]              = texture_format( GL_RG32F,               GL_RG,               8, GL_FLOAT,                  "GL_RG32F" );            
    _texture_formats[GL_RGB32F]             = texture_format( GL_RGB32F,              GL_RGB,              12,GL_FLOAT,                  "GL_RGB32F" );           
    _texture_formats[GL_RGBA32F]            = texture_format( GL_RGBA32F,             GL_RGBA,             16,GL_FLOAT,                  "GL_RGBA32F" );          
    _texture_formats[GL_R11F_G11F_B10F]     = texture_format( GL_R11F_G11F_B10F,      GL_RGB,              4, GL_FLOAT,                  "GL_R11F_G11F_B10F" );   
    _texture_formats[GL_RGB9_E5]            = texture_format( GL_RGB9_E5,             GL_RGB,              4, GL_FLOAT,                  "GL_RGB9_E5" );          
    _texture_formats[GL_R8I]                = texture_format( GL_R8I,                 GL_RED,              1, GL_BYTE,                   "GL_R8I" );              
    _texture_formats[GL_R8UI]               = texture_format( GL_R8UI,                GL_RED,              1, GL_UNSIGNED_BYTE,          "GL_R8UI" );             
    _texture_formats[GL_R16I]               = texture_format( GL_R16I,                GL_RED,              2, GL_SHORT,                  "GL_R16I" );             
    _texture_formats[GL_R16UI]              = texture_format( GL_R16UI,               GL_RED,              2, GL_UNSIGNED_SHORT,         "GL_R16UI" );            
    _texture_formats[GL_R32I]               = texture_format( GL_R32I,                GL_RED,              4, GL_INT,                    "GL_R32I" );             
    _texture_formats[GL_R32UI]              = texture_format( GL_R32UI,               GL_RED,              4, GL_UNSIGNED_INT,           "GL_R32UI" );            
    _texture_formats[GL_RG8I]               = texture_format( GL_RG8I,                GL_RG,               2, GL_BYTE,                   "GL_RG8I" );             
    _texture_formats[GL_RG8UI]              = texture_format( GL_RG8UI,               GL_RG,               2, GL_UNSIGNED_BYTE,          "GL_RG8UI" );            
    _texture_formats[GL_RG16I]              = texture_format( GL_RG16I,               GL_RG,               4, GL_SHORT,                  "GL_RG16I" );            
    _texture_formats[GL_RG16UI]             = texture_format( GL_RG16UI,              GL_RG,               4, GL_UNSIGNED_SHORT,         "GL_RG16UI" );           
    _texture_formats[GL_RG32I]              = texture_format( GL_RG32I,               GL_RG,               8, GL_INT,                    "GL_RG32I" );            
    _texture_formats[GL_RG32UI]             = texture_format( GL_RG32UI,              GL_RG,               8, GL_UNSIGNED_INT,           "GL_RG32UI" );           
    _texture_formats[GL_RGB8I]              = texture_format( GL_RGB8I,               GL_RGB,              3, GL_BYTE,                   "GL_RGB8I" );            
    _texture_formats[GL_RGB8UI]             = texture_format( GL_RGB8UI,              GL_RGB,              3, GL_UNSIGNED_BYTE,          "GL_RGB8UI" );           
    _texture_formats[GL_RGB16I]             = texture_format( GL_RGB16I,              GL_RGB,              6, GL_SHORT,                  "GL_RGB16I" );           
    _texture_formats[GL_RGB16UI]            = texture_format( GL_RGB16UI,             GL_RGB,              6, GL_UNSIGNED_SHORT,         "GL_RGB16UI" );          
    _texture_formats[GL_RGB32I]             = texture_format( GL_RGB32I,              GL_RGB,              12,GL_INT,                    "GL_RGB32I" );           
    _texture_formats[GL_RGB32UI]            = texture_format( GL_RGB32UI,             GL_RGB,              12,GL_UNSIGNED_INT,           "GL_RGB32UI" );          
    _texture_formats[GL_RGBA8I]             = texture_format( GL_RGBA8I,              GL_RGBA,             4, GL_BYTE,                   "GL_RGBA8I" );           
    _texture_formats[GL_RGBA8UI]            = texture_format( GL_RGBA8UI,             GL_RGBA,             4, GL_UNSIGNED_BYTE,          "GL_RGBA8UI" );          
    _texture_formats[GL_RGBA16I]            = texture_format( GL_RGBA16I,             GL_RGBA,             8, GL_SHORT,                  "GL_RGBA16I" );          
    _texture_formats[GL_RGBA16UI]           = texture_format( GL_RGBA16UI,            GL_RGBA,             8, GL_UNSIGNED_SHORT,         "GL_RGBA16UI" );         
    _texture_formats[GL_RGBA32I]            = texture_format( GL_RGBA32I,             GL_RGBA,             16,GL_INT,                    "GL_RGBA32I" );          
    _texture_formats[GL_RGBA32UI]           = texture_format( GL_RGBA32UI,            GL_RGBA,             16,GL_UNSIGNED_INT,           "GL_RGBA32UI" );         
    _texture_formats[GL_DEPTH_COMPONENT16]  = texture_format( GL_DEPTH_COMPONENT16,   GL_DEPTH_COMPONENT,   2,GL_FLOAT,                  "GL_DEPTH_COMPONENT16" );
    _texture_formats[GL_DEPTH_COMPONENT24]  = texture_format( GL_DEPTH_COMPONENT24,   GL_DEPTH_COMPONENT,   3,GL_FLOAT,                  "GL_DEPTH_COMPONENT24" );
    _texture_formats[GL_DEPTH_COMPONENT32]  = texture_format( GL_DEPTH_COMPONENT32,   GL_DEPTH_COMPONENT,   4,GL_FLOAT,                  "GL_DEPTH_COMPONENT32" );
    _texture_formats[GL_DEPTH_COMPONENT32F] = texture_format( GL_DEPTH_COMPONENT32F,  GL_DEPTH_COMPONENT,   4,GL_FLOAT,                  "GL_DEPTH_COMPONENT32F" );
    _texture_formats[GL_DEPTH24_STENCIL8]   = texture_format( GL_DEPTH24_STENCIL8,    GL_DEPTH_STENCIL,     4,GL_FLOAT,                  "GL_DEPTH24_STENCIL8" ); 
    _texture_formats[GL_DEPTH32F_STENCIL8]  = texture_format( GL_DEPTH32F_STENCIL8,   GL_DEPTH_STENCIL,     5,GL_FLOAT,                  "GL_DEPTH32F_STENCIL8" );
  }
}


} } // namespace gpucast / namespace gl
