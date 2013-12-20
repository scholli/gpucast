/********************************************************************************
*
* Copyright (C) 2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : geode.hpp
*  project    : glpp
*  description:
*
********************************************************************************/
#ifndef GPUCAST_GL_GEODE_HPP
#define GPUCAST_GL_GEODE_HPP

#include <list>
#include <vector>

#include <gpucast/gl/glpp.hpp>
#include <gpucast/gl/math/vec4.hpp>

#include <gpucast/gl/arraybuffer.hpp>
#include <gpucast/gl/elementarraybuffer.hpp>
#include <gpucast/gl/vertexarrayobject.hpp>
#include <gpucast/gl/util/material.hpp>
#include <gpucast/gl/graph/group.hpp>

namespace gpucast { namespace gl {

  class GPUCAST_GL geode : public node
  {
    public : // typedef / enums

      // additional information for file input
      enum attribute_type { vertex,
                            color,
                            normal,
                            texcoord,
                            unspecified
                          };

      struct attribute_buffer 
      {
        std::size_t                     location;
        std::vector<vec4f>              clientbuffer;
        std::shared_ptr<arraybuffer>  buffer;
        attribute_type                  type; 
      };

    public : // c'tor

      geode          ();
      virtual ~geode ();

    public :

      virtual void    visit                   ( visitor const& v );
      void            draw                    ( ) const;
      void            draw_elements           ( GLenum      mode, 
                                                std::size_t count,
                                                std::size_t start_index ) const;

      virtual void    compute_bbox            ();


      void            add_attribute_buffer    ( std::size_t               location,
                                                std::vector<vec4f> const& data, 
                                                attribute_type            type = unspecified );

      // get and set information on attributes
      std::size_t     attributes              () const;

      void            set_attribute_location  ( std::size_t index, std::size_t location );
      std::size_t     get_attribute_location  ( std::size_t index ) const;

      void            set_attribute_type      ( std::size_t index, attribute_type type);
      attribute_type  get_attribute_type      ( std::size_t index ) const;

      // get and set elementarray
      void            set_indexbuffer         ( std::vector<int> const& buf );

      void            set_mode                ( GLenum mode );

      void            set_material            ( material const& );
      material const& get_material            ( ) const;

    private :

      std::vector<attribute_buffer>           _attributes;

      std::shared_ptr<elementarraybuffer>     _indices;
      vertexarrayobject                       _vao;
      std::size_t                             _number_of_indices;
      
      GLenum                                  _mode;
      material                                _material;
  };

} } // namespace gpucast / namespace gl

#endif 