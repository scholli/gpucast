/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : ocnode.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_OCNODE_HPP
#define GPUCAST_OCNODE_HPP

// header, system

// header, external
#include <memory>

#include <gpucast/math/axis_aligned_boundingbox.hpp>

// header, project
#include <gpucast/volume/gpucast.hpp>

#include <gpucast/volume/isosurface/octree/node.hpp>


namespace gpucast {

class GPUCAST_VOLUME ocnode : public node
{
public : // typedefs / enums

  typedef gpucast::math::interval<value_type>       interval_t;

  typedef std::vector<face_ptr>           face_container;
  typedef face_container::iterator        face_iterator;
  typedef face_container::const_iterator  const_face_iterator;

public : // c'tor / d'tor

  ocnode      ();

  ocnode      ( pointer const&           parent,
                std::size_t              depth,
                std::size_t              id );

  ocnode      ( pointer const&          parent,
                std::size_t             depth, 
                boundingbox_type const& size,
                std::size_t             id );

  ~ocnode     ();

public : // methods

  /* virtual */ boundingbox_type const& boundingbox         ( ) const;
  void                                  boundingbox         ( boundingbox_type const& bb );

  face_iterator                         face_begin          ();
  face_iterator                         face_end            ();

  const_face_iterator                   face_begin          () const;
  const_face_iterator                   face_end            () const;

  void                                  add_face            ( face_ptr const& );
  void                                  clear_faces         ();

  interval_t const&                     range               () const;
  void                                  range               ( interval_t const& );

  bool                                  contains_outer_face () const;
  void                                  contains_outer_face ( bool );

  bool                                  empty               () const;
  std::size_t                           faces               () const;

  /* virtual */ void                    compute_bbox_from_children ();
  /* virtual */ void                    compute_bbox_from_data     ();

  /* virtual */ value_type              volume              () const;
  /* virtual */ value_type              surface             () const;

  /* virtual */ void                    accept              ( nodevisitor const& visitor );
  /* virtual */ void                    draw                ( gpucast::math::matrix4x4<float> const& mvp );

  /* virtual */ void                    print               ( std::ostream& os ) const;

private : // auxilliary methods

private : // attributes

  boundingbox_type                    _bbox;
  bool                                _contains_outer_face;

  interval_t                          _range;
  face_container                      _faces;

  std::size_t                         _id;
};

typedef std::shared_ptr<ocnode> ocnode_ptr;

std::ostream& operator<<(std::ostream& os, ocnode const& node);

} // namespace gpucast

#endif // GPUCAST_OCNODE_HPP