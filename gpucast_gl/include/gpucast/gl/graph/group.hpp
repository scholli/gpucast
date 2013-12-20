/********************************************************************************
*
* Copyright (C) 2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : group.hpp
*  project    : glpp
*  description:
*
********************************************************************************/
#ifndef GPUCAST_GL_GROUP_HPP
#define GPUCAST_GL_GROUP_HPP

#include <set>

#include <boost/shared_ptr.hpp>

#include <gpucast/gl/glpp.hpp>
#include <gpucast/gl/math/matrix4x4.hpp>
#include <gpucast/gl/graph/node.hpp>

#include <gpucast/math/axis_aligned_boundingbox.hpp>

namespace gpucast { namespace gl {

  class GPUCAST_GL group : public node
  {
    public : 

      typedef gpucast::math::axis_aligned_boundingbox<vec3f>       bbox_t;
      typedef std::shared_ptr<node>         node_ptr_t;
      typedef std::set<node_ptr_t>          container_t;

      typedef container_t::iterator         iterator;
      typedef container_t::const_iterator   const_iterator;

    public :

      group                     ( );
      virtual ~group            ( );

    public :

      virtual void            visit         ( visitor const& v );
      virtual void            compute_bbox  ();

      // group methods
      void                    add           ( node_ptr_t child );
      void                    remove        ( node_ptr_t child );
      std::size_t             children      () const;

      // iterator interface
      iterator                begin         ();
      iterator                end           ();
      const_iterator          begin         () const;
      const_iterator          end           () const;

      // transform methods
      void                    set_transform ( matrix4f const&   matrix );

      void                    translate     ( vec3f const& translation);

      void                    rotate        ( float        alpha, 
                                              vec3f const& axis);

      void                    scale         ( vec3f const& scale );

    private :
      
      matrix4f                _matrix;
      container_t             _children;
  };

} } // namespace gpucast / namespace gl

#endif 
