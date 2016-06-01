/********************************************************************************
*
* Copyright (C) 2016 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : trimdomain_serializer_contour_map_kd.cpp
*  project    : gpucast
*  description:
*
********************************************************************************/

// header i/f
#include "gpucast/core/trimdomain_serializer_contour_map_kd.hpp"

// header, system

// header, project

namespace gpucast {

  /////////////////////////////////////////////////////////////////////////////
  trimdomain_serializer::address_type
    trimdomain_serializer_contour_map_kd::serialize(trimdomain_ptr const&                                     input_domain,
    kd_split_strategy                                         kdtree_generation,
    trim_kd_serialization&                                    serialization,
    bool                                                      texture_classification_enabled,
    unsigned                                                  texture_classification_resolution) const
  {
    typedef gpucast::math::domain::contour_map_kd<typename beziersurface::curve_point_type::value_type> contour_map_type;
    typedef contour_map_type::kdtree_type::kdnode_ptr kdnode_ptr;
    assert(serialization.partition.size() < std::numeric_limits<address_type>::max());

    // if already in buffer -> return index
    if (serialization.domain_index_map.count(input_domain)) {
      return serialization.domain_index_map.find(input_domain)->second;
    }

    address_type partition_base_index = serialization.partition.size();
    contour_map_type kdtree(kdtree_generation, texture_classification_enabled, texture_classification_resolution);

    // fill loops into contour map
    std::for_each(input_domain->loops().begin(), input_domain->loops().end(), std::bind(&contour_map_type::add, &kdtree, std::placeholders::_1));
    kdtree.initialize();

    // generate fast pre-classification texture
    address_type classification_id = 0;
    if (texture_classification_enabled) {
      classification_id = trimdomain_serializer::serialize(input_domain, serialization.preclassification, texture_classification_resolution);
    }

    float_type umin = kdtree.bounds().min[point_type::u];
    float_type umax = kdtree.bounds().max[point_type::u];
    float_type vmin = kdtree.bounds().min[point_type::v];
    float_type vmax = kdtree.bounds().max[point_type::v];

    // serialize tree
    std::vector<kdnode_ptr> nodes;
    kdtree.partition().root->serialize_dfs(nodes);

    // retrieve indices to pointer mapping
    std::unordered_map<kdnode_ptr, unsigned> node_offset_map;
    unsigned offset = 0;
    for (kdnode_ptr const& node : nodes) {
      node_offset_map.insert(std::make_pair(node, offset++));
    }

    // reserve enough memory
    const std::size_t header_size = 3;
    serialization.partition.resize(partition_base_index + header_size);
    serialization.partition[partition_base_index] = math::vec4f(unsigned_bits_as_float(nodes.size() != 0),
      unsigned_bits_as_float(classification_id),
      unsigned_bits_as_float(kdtree.pre_classification().width()),
      unsigned_bits_as_float(kdtree.pre_classification().height())); // base entry
    // second entry is size of partition
    serialization.partition[partition_base_index + 1] = math::vec4f(umin, umax, vmin, vmax);

    // third entry is size of domain
    serialization.partition[partition_base_index + 2] = math::vec4f(input_domain->nurbsdomain().min[point_type::u],
      input_domain->nurbsdomain().max[point_type::u],
      input_domain->nurbsdomain().min[point_type::v],
      input_domain->nurbsdomain().max[point_type::v]);

    assert(nodes.size() < std::numeric_limits<address_type>::max());
    for (kdnode_ptr const& node : nodes)
    {
      assert(node->overlapping_segments.size() < std::numeric_limits<char>::max());
      assert(node->is_leaf() < std::numeric_limits<char>::max());
      assert(node->split_direction < std::numeric_limits<char>::max());

      address_type node_base_info = uint4ToUInt(unsigned(node->is_leaf()),
        node->parity,
        unsigned(node->split_direction),
        unsigned(node->overlapping_segments.size()));
      if (node->is_leaf()) {
        address_type contourlist_base_index = serialization.contourlist.size();
        serialization.partition.push_back(math::vec4f(unsigned_bits_as_float(node_base_info),
          unsigned_bits_as_float(contourlist_base_index),
          0,  // spare
          0   // spare
          ));

        // serialize segments in leaf
        for (auto const& contour_segment : node->overlapping_segments)
        {
          address_type ncurves_uincreasing = uint2x16ToUInt(explicit_type_conversion<size_t, address_type>(contour_segment->size()),
            contour_segment->is_increasing(point_type::u));

          address_type curvelist_id = serialize_contour_segment(contour_segment, serialization);

          serialization.contourlist.push_back(math::vec4f(unsigned_bits_as_float(ncurves_uincreasing),
            unsigned_bits_as_float(curvelist_id),
            unsigned_bits_as_float(float2_to_unsigned(contour_segment->bbox().min[point_type::u], contour_segment->bbox().min[point_type::v])),
            unsigned_bits_as_float(float2_to_unsigned(contour_segment->bbox().max[point_type::u], contour_segment->bbox().max[point_type::v]))));
        }

      }
      else { // node is no leaf -> just add split value and children
        address_type less_id = partition_base_index + header_size + node_offset_map[node->child_less];
        address_type greater_id = partition_base_index + header_size + node_offset_map[node->child_greater];

        serialization.partition.push_back(math::vec4f(unsigned_bits_as_float(node_base_info),
          explicit_type_conversion<double, float>(node->split_value),
          unsigned_bits_as_float(less_id),
          unsigned_bits_as_float(greater_id)));
      }
    }

    // store domain_ptr to index mapping for later reference
    serialization.domain_index_map.insert(std::make_pair(input_domain, partition_base_index));

    // make sure buffers are still in range of address_type
    if (serialization.partition.size() >= std::numeric_limits<address_type>::max())
    {
      throw std::runtime_error("Address exceeds maximum of addressable memory");
    }

    return partition_base_index;
  }

  /////////////////////////////////////////////////////////////////////////////
  trimdomain_serializer::address_type
    trimdomain_serializer_contour_map_kd::serialize_contour_segment(contour_segment_ptr const& contour_segment,
    trim_kd_serialization&     serialization) const
  {
    if (serialization.contour_index_map.count(contour_segment))
    {
      return serialization.contour_index_map[contour_segment];
    }

    address_type contour_segment_index = serialization.curvelist.size();

    for (auto i_cptr = contour_segment->begin(); i_cptr != contour_segment->end(); ++i_cptr)
    {
      curve_ptr curve = *i_cptr;

      address_type curveid = 0;
      if (!curve->is_constant(point_type::u) &&
        !curve->is_constant(point_type::v)) // linear curves do not have to be tested
      {
        curveid = trimdomain_serializer::serialize(curve, serialization.curve_index_map, serialization.pointdata);
      }

      bbox_type bbox;
      curve->bbox_simple(bbox);

      serialization.curvelist.push_back(math::vec4f(bbox.min[point_type::v],
        bbox.max[point_type::v],
        bbox.min[point_type::u],
        bbox.max[point_type::u]));

      serialization.curvedata.push_back(float_type(unsigned_bits_as_float(uint8_24ToUInt(curve->order(), curveid))));
    }

    return contour_segment_index;
  }


} // namespace gpucast