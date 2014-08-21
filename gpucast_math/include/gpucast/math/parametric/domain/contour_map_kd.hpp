/********************************************************************************
*
* Copyright (C) 2013 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : contour_map_kd.hpp
*
*  description:
*
********************************************************************************/
#ifndef GPUCAST_MATH_CONTOUR_MAP_KD_HPP
#define GPUCAST_MATH_CONTOUR_MAP_KD_HPP

// includes, system

// includes, project
#include <gpucast/math/parametric/domain/contour_map_base.hpp>

namespace gpucast { namespace math {

template <typename value_t>
class contour_map_kd : public contour_map_base<value_t>
{
  public : // typedef / enums

    typedef contour_map_base<value_t>              base_type;
    using typename contour_map_base<value_t>::interval_type;
    using typename contour_map_base<value_t>::contour_segment_container;
    using typename contour_map_base<value_t>::value_type;
    using typename contour_map_base<value_t>::point_type;
    using typename contour_map_base<value_t>::bbox_type;
    using typename contour_map_base<value_t>::contour_segment_ptr;
    using typename contour_map_base<value_t>::contour_type;
    using typename contour_map_base<value_t>::contour_segment_type;

    struct contour_cell
    {
      void print ( std::ostream& os ) const;

      contour_segment_container overlapping_segments;
      interval_type             interval_u;
      interval_type             interval_v;
      bool                      inside;
    };

    struct kdnode
    {
      kdnode ( value_type const& s, 
               typename point_type::coordinate_type const& d, 
               contour_cell* c,
               kdnode* l, 
               kdnode* m )
        : split_value(s), split_dimension(d), cell(c), less (l), more (m) {};

      bool is_child () const { return cell != nullptr; };

      value_type                           split_value;
      typename point_type::coordinate_type split_dimension;
      contour_cell*                        cell;
      kdnode*                              less;
      kdnode*                              more;
    };

  public : // c'tor / d'tor

    contour_map_kd   ();
    virtual ~contour_map_kd  ();

  public : // methods

    /* virtual */ void                initialize       ();

    void                              destroy          ( kdnode* n );

    kdnode*                           create           ( bbox_type const& bounds, 
                                                         std::vector<contour_cell> const& cells );

    kdnode*                           split            ( bbox_type const& bounds, 
                                                         typename point_type::coordinate_type const& dim, 
                                                         std::set<value_type> const& candidates,
                                                         std::vector<contour_cell> const& cells );

    std::set<value_t>                 split_candidates ( bbox_type const& bounds, 
                                                         typename point_type::coordinate_type const& dim, 
                                                         std::vector<contour_cell> const& cells );


    // stream output of domain
    virtual void                      print                         ( std::ostream& os ) const;

  protected : // methods

  private : // internal/auxilliary methods

    void _insert_contour_segment (contour_segment_ptr const&);

  protected : // attributes

    std::vector<contour_cell>     _cells;

    kdnode*                       _root;
};


template <typename value_t>
std::ostream& operator<<(std::ostream& os,  gpucast::math::contour_map_kd<value_t> const& rhs);

} } // namespace gpucast / namespace math

#include "contour_map_kd_impl.hpp"

#endif // GPUCAST_MATH_RECTANGULAR_MAP_HPP
