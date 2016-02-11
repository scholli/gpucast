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
#ifndef GPUCAST_MATH_CONTOUR_MAP_LOOP_LIST_HPP
#define GPUCAST_MATH_CONTOUR_MAP_LOOP_LIST_HPP

// includes, system

// includes, project
#include <gpucast/math/parametric/domain/partition/monotonic_contour/contour_map_base.hpp>

namespace gpucast {
  namespace math {
    namespace domain {

      template <typename value_t>
      class contour_map_loop_list : public contour_map_base<value_t>
      {
      public: // typedef / enums

        typedef contour_map_base<value_t> base_type;

        using typename base_type::interval_type;
        using typename base_type::value_type;
        using typename base_type::point_type;
        using typename base_type::bbox_type;
        using typename base_type::contour_segment_type;
        using typename base_type::contour_segment_ptr;
        using typename base_type::contour_segment_container;
        using typename base_type::contour_container;
        using typename base_type::contour_type;
        using typename base_type::contour_ptr;
        
        struct trimloop {
          contour_ptr           contour;
          std::vector<trimloop> children;
        };

      public: // c'tor / d'tor

      public: // methods

        /*virtual*/ bool initialize() override;
        /*virtual*/ void print(std::ostream& os) const override;

        trimloop const&  root() const;

        // returns if left of segment is inside or outside of the domain
        unsigned         parity(contour_segment_ptr const& segment) const;

        // returns in how many segment bounding boxes is this segment
        unsigned         priority(contour_segment_ptr const& segment) const;

      protected: // methods

      private: // internal/auxilliary methods

        contour_ptr const&  _determine_outer_loop (contour_container const& in_loops);
        bool                _is_child (contour_ptr const& parent, contour_ptr const& child, contour_container const& other_loops);

        void                _determine_contour_segments_parity (); 
        void                _determine_contour_segments_priority ();

      protected: // attributes

        trimloop                                          _outer_loop;

        std::unordered_map<contour_segment_ptr, unsigned> _segment_parity_classification;
        std::unordered_map<contour_segment_ptr, unsigned> _segment_priority_classification;

      };

      template <typename value_t>
      std::ostream& operator<<(std::ostream& os, gpucast::math::domain::contour_map_loop_list<value_t> const& rhs);

    } // namespace domain
  } // namespace math
} // namespace gpucast 

#include "contour_map_loop_list_impl.hpp"

#endif // GPUCAST_MATH_CONTOUR_MAP_LOOP_LIST_HPP
