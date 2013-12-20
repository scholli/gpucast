/********************************************************************************
*
* Copyright (C) 2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : node.hpp
*  project    : glpp
*  description:
*
********************************************************************************/
#ifndef GPUCAST_GL_NODE_HPP
#define GPUCAST_GL_NODE_HPP

#include <string>

#include <gpucast/gl/glpp.hpp>

#include <gpucast/math/axis_aligned_boundingbox.hpp>
#include <gpucast/math/parametric/point.hpp>

namespace gpucast { namespace gl {

  class visitor;

  class GPUCAST_GL node 
  {
    public : // enums / typedefs

      typedef gpucast::math::axis_aligned_boundingbox<gpucast::math::point3f> bbox_t;

    public : // c'tor / d'tor

      node          ();
      virtual ~node ();
      
    public : // methods

      virtual void          visit         ( visitor const& v ) =0;
      virtual void          compute_bbox  () =0;

      bbox_t const&         bbox          () const;
      
      void                  name          ( std::string const& );
      std::string const&    name          ( ) const;


    protected : // attributes

      bbox_t                _bbox;

    private : 

      std::string           _name;

  };

} } // namespace gpucast / namespace gl

#endif 