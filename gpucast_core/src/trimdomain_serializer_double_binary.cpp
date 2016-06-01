/********************************************************************************
*
* Copyright (C) 2016 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : trimdomain_serializer_double_binary.cpp
*  project    : gpucast
*  description:
*
********************************************************************************/

// header i/f
#include "gpucast/core/trimdomain_serializer_double_binary.hpp"

// header, system

// header, project

namespace gpucast {

  /////////////////////////////////////////////////////////////////////////////
  trimdomain_serializer::address_type
    trimdomain_serializer_double_binary::serialize(trimdomain_ptr const&              input_domain,
    trim_double_binary_serialization&  serialization,
    bool                               pre_classification_enabled,
    unsigned                           pre_classification_resolution) const
  {
    // if already in buffer -> return index
    if (serialization.domain_index_map.count(input_domain))
    {
      return serialization.domain_index_map.find(input_domain)->second;
    }

    // copy curves to double binary partition
    trimdomain::curve_container curves = input_domain->curves();
    gpucast::math::domain::partition<trimdomain_serializer::point_type>  input_partition(curves.begin(), curves.end());

    // initialize partition
    input_partition.initialize();

    // entry point in texture
    assert(serialization.partition.size() < std::numeric_limits<address_type>::max());
    assert(input_partition.size() < std::numeric_limits<address_type>::max());

    address_type partition_index = explicit_type_conversion<std::size_t, unsigned>(serialization.partition.size());
    address_type vintervals = explicit_type_conversion<std::size_t, unsigned>(input_partition.size());

    float_type vmin = explicit_type_conversion<double, float>(input_partition.get_vertical_interval().minimum());
    float_type vmax = explicit_type_conversion<double, float>(input_partition.get_vertical_interval().maximum());
    float_type umin = explicit_type_conversion<double, float>(input_partition.get_horizontal_interval().minimum());
    float_type umax = explicit_type_conversion<double, float>(input_partition.get_horizontal_interval().maximum());

    // generate fast pre-classification texture
    address_type classification_id = 0;
    if (pre_classification_enabled) {
      classification_id = trimdomain_serializer::serialize(input_domain, serialization.preclassification, pre_classification_resolution);
    }

    serialization.partition.push_back(math::vec4f(unsigned_bits_as_float(vintervals),
      unsigned_bits_as_float(classification_id),
      unsigned_bits_as_float(pre_classification_resolution),
      unsigned_bits_as_float(pre_classification_resolution)
      ));

    serialization.partition.push_back(math::vec4f(umin, umax, vmin, vmax));

    serialization.partition.push_back(math::vec4f(input_domain->nurbsdomain().min[point_type::u],
      input_domain->nurbsdomain().max[point_type::u],
      input_domain->nurbsdomain().min[point_type::v],
      input_domain->nurbsdomain().max[point_type::v]));

    for (auto const& v : input_partition) {
      assert(serialization.celldata.size() < std::numeric_limits<address_type>::max());
      assert(v->size() < std::numeric_limits<address_type>::max());

      serialization.partition.push_back(math::vec4f(explicit_type_conversion<double, float>(v->get_vertical_interval().minimum()),
        explicit_type_conversion<double, float>(v->get_vertical_interval().maximum()),
        unsigned_bits_as_float(address_type(serialization.celldata.size())),
        unsigned_bits_as_float(address_type(v->size()))));

      serialization.celldata.push_back(math::vec4f(explicit_type_conversion<double, float>(v->get_horizontal_interval().minimum()),
        explicit_type_conversion<double, float>(v->get_horizontal_interval().maximum()),
        unsigned_bits_as_float(address_type(v->size())),
        0));

      for (gpucast::math::domain::partition<gpucast::math::point2d>::cell_ptr_type const& c : *v)
      {
        assert(c->intersections() < std::numeric_limits<address_type>::max());
        assert(serialization.curvelist.size() < std::numeric_limits<address_type>::max());
        assert(c->size() < std::numeric_limits<address_type>::max());

        serialization.celldata.push_back(math::vec4f(explicit_type_conversion<double, float>(c->get_horizontal_interval().minimum()),
          explicit_type_conversion<double, float>(c->get_horizontal_interval().maximum()),
          unsigned_bits_as_float(address_type(c->intersections())),
          unsigned_bits_as_float(address_type(serialization.curvelist.size()))));

        serialization.curvelist.push_back(math::vec4f(unsigned_bits_as_float(address_type(c->size())), 0, 0, 0));

        for (gpucast::math::domain::partition<gpucast::math::point2d>::curve_segment_ptr const& cv : *c)
        {
          address_type curve_index = trimdomain_serializer::serialize(cv->curve(), serialization.curve_index_map, serialization.curvedata);
          //address_type order_and_u_increase = cv->curve()->is_increasing(point_type::u) ? explicit_type_conversion<size_t, int>(cv->curve()->order()) : -explicit_type_conversion<size_t, int>(cv->curve()->order());
          int order_and_u_increase = cv->curve()->is_increasing(point_type::u) ? int(cv->curve()->order()) : -1 * int(cv->curve()->order());

          serialization.curvelist.push_back(math::vec4f(unsigned_bits_as_float(curve_index),
            unsigned_bits_as_float(order_and_u_increase),
            explicit_type_conversion<double, float>(cv->tmin()),
            explicit_type_conversion<double, float>(cv->tmax())));
        }
      }
    }

    serialization.domain_index_map.insert(std::make_pair(input_domain, partition_index));

    // make sure buffers are still in range of address_type
    if (serialization.partition.size() >= std::numeric_limits<address_type>::max() ||
      serialization.celldata.size() >= std::numeric_limits<address_type>::max() ||
      serialization.curvelist.size() >= std::numeric_limits<address_type>::max() ||
      serialization.curvedata.size() >= std::numeric_limits<address_type>::max() ||
      input_partition.size() >= std::numeric_limits<address_type>::max())
    {
      throw std::runtime_error("Address exceeds maximum of addressable memory");
    }

    return partition_index;
  }


} // namespace gpucast