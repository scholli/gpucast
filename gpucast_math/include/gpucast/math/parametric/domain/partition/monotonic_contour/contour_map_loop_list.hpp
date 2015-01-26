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
        using typename contour_map_base<value_t>::interval_type;
        using typename contour_map_base<value_t>::contour_segment_container;
        using typename contour_map_base<value_t>::value_type;
        using typename contour_map_base<value_t>::point_type;
        using typename contour_map_base<value_t>::bbox_type;
        using typename contour_map_base<value_t>::contour_segment_ptr;
        using typename contour_map_base<value_t>::contour_type;
        using typename contour_map_base<value_t>::contour_segment_type;

      public: // c'tor / d'tor

      public: // methods

        /*virtual*/ void initialize() override;
        /*virtual*/ void print(std::ostream& os) const override;

        trimloop const& root() const;

      protected: // methods

      private: // internal/auxilliary methods

        contour_ptr const& _determine_outer_loop (contour_container const& in_loops);
        bool _is_child(contour_ptr const& parent, contour_ptr const& child, contour_container const& other_loops);

      protected: // attributes

        struct trimloop {
          contour_ptr           contour;
          std::vector<trimloop> children;
        };

        trimloop _outer_loop;

      };

      template <typename value_t>
      std::ostream& operator<<(std::ostream& os, gpucast::math::domain::contour_map_loop_list<value_t> const& rhs);

    } // namespace domain
  } // namespace math
} // namespace gpucast 

#include "contour_map_loop_list_impl.hpp"

#endif // GPUCAST_MATH_CONTOUR_MAP_LOOP_LIST_HPP
