/********************************************************************************
*
* Copyright (C) 2016 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : trimdomain_serializer_contour_map_binary.cpp
*  project    : gpucast
*  description:
*
********************************************************************************/

// header i/f
#include "gpucast/core/trimdomain_serializer_contour_map_binary.hpp"

// header, system

// header, project

namespace gpucast {

  /////////////////////////////////////////////////////////////////////////////
  trimdomain_serializer::address_type
    trimdomain_serializer_contour_map_binary::serialize(trimdomain_ptr const&              input_domain,
    trim_contour_binary_serialization& serialization,
    bool                               pre_classification_enabled,
    unsigned                           pre_classification_resolution) const
  {
    typedef gpucast::math::domain::contour_map_binary<beziersurface::curve_point_type::value_type> contour_map_type;
    assert(serialization.partition.size() < std::numeric_limits<address_type>::max());

    // if already in buffer -> return index
    if (serialization.domain_index_map.count(input_domain))
    {
      return serialization.domain_index_map.find(input_domain)->second;
    }

    address_type partition_index = explicit_type_conversion<std::size_t, address_type>(serialization.partition.size());
    contour_map_type map;

    // fill loops into contour map
    std::for_each(input_domain->loops().begin(), input_domain->loops().end(), std::bind(&contour_map_type::add, &map, std::placeholders::_1));
    map.initialize();

    float_type umin = explicit_type_conversion<double, float>(map.bounds().min[point_type::u]);
    float_type umax = explicit_type_conversion<double, float>(map.bounds().max[point_type::u]);
    float_type vmin = explicit_type_conversion<double, float>(map.bounds().min[point_type::v]);
    float_type vmax = explicit_type_conversion<double, float>(map.bounds().max[point_type::v]);

    // generate fast pre-classification texture
    address_type classification_id = 0;
    if (pre_classification_enabled) {
      classification_id = trimdomain_serializer::serialize(input_domain, serialization.preclassification, pre_classification_resolution);
    }

    assert(map.partition().size() < std::numeric_limits<address_type>::max());
    address_type vintervals = explicit_type_conversion<std::size_t, address_type>(map.partition().size());

    const std::size_t header_size = 3;
    serialization.partition.resize(partition_index + header_size + map.partition().size());

    // information header
    serialization.partition[partition_index] = math::vec4f(unsigned_bits_as_float(vintervals),
      unsigned_bits_as_float(classification_id),
      unsigned_bits_as_float(pre_classification_resolution),
      unsigned_bits_as_float(pre_classification_resolution));
    // outer loop bounds
    serialization.partition[partition_index + 1] = math::vec4f(umin, umax, vmin, vmax);

    // total domain bounds
    serialization.partition[partition_index + 2] = math::vec4f(explicit_type_conversion<double, float>(input_domain->nurbsdomain().min[point_type::u]),
      explicit_type_conversion<double, float>(input_domain->nurbsdomain().max[point_type::u]),
      explicit_type_conversion<double, float>(input_domain->nurbsdomain().min[point_type::v]),
      explicit_type_conversion<double, float>(input_domain->nurbsdomain().max[point_type::v]));
    std::size_t vindex = partition_index + header_size;

    for (auto const& vinterval : map.partition())
    {
      assert(vinterval.cells.size() < std::numeric_limits<address_type>::max());

      address_type uid = explicit_type_conversion<size_t, address_type>(serialization.partition.size());
      address_type ucells = explicit_type_conversion<size_t, address_type>(vinterval.cells.size());

      serialization.partition[vindex++] = math::vec4f(explicit_type_conversion<double, float>(vinterval.interval_v.minimum()),
        explicit_type_conversion<double, float>(vinterval.interval_v.maximum()),
        unsigned_bits_as_float(uid),
        unsigned_bits_as_float(ucells));
      serialization.partition.push_back(math::vec4f(explicit_type_conversion<double, float>(vinterval.interval_u.minimum()),
        explicit_type_conversion<double, float>(vinterval.interval_u.maximum()),
        0, // spare
        0  // spare 
        ));

      for (auto const& cell : vinterval.cells)
      {
        address_type contourlist_id = explicit_type_conversion<std::size_t, unsigned>(serialization.contourlist.size());
        address_type type_and_contours = uint4ToUInt(0, cell.inside, explicit_type_conversion<size_t, address_type>(cell.overlapping_segments.size()), 0);

        serialization.partition.push_back(math::vec4f(explicit_type_conversion<double, float>(cell.interval_u.minimum()),
          explicit_type_conversion<double, float>(cell.interval_u.maximum()),
          unsigned_bits_as_float(type_and_contours),
          unsigned_bits_as_float(contourlist_id)));

        for (auto const& contour_segment : cell.overlapping_segments)
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
    }

    // store domain_ptr to index mapping for later reference
    serialization.domain_index_map.insert(std::make_pair(input_domain, partition_index));

    // make sure buffers are still in range of address_type
    if (serialization.partition.size() >= std::numeric_limits<address_type>::max())
    {
      throw std::runtime_error("Address exceeds maximum of addressable memory");
    }

    return partition_index;
  }

  /////////////////////////////////////////////////////////////////////////////
  trimdomain_serializer::address_type
    trimdomain_serializer_contour_map_binary::serialize_contour_segment(contour_segment_ptr const&         contour_segment,
    trim_contour_binary_serialization& serialization) const
  {
    if (serialization.contour_index_map.count(contour_segment))
    {
      return serialization.contour_index_map[contour_segment];
    }

    address_type contour_segment_index = explicit_type_conversion<size_t, address_type>(serialization.curvelist.size());

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

      serialization.curvelist.push_back(math::vec4f(explicit_type_conversion<double, float>(bbox.min[point_type::v]),
        explicit_type_conversion<double, float>(bbox.max[point_type::v]),
        explicit_type_conversion<double, float>(bbox.min[point_type::u]),
        explicit_type_conversion<double, float>(bbox.max[point_type::u])));

      serialization.curvedata.push_back(float_type(unsigned_bits_as_float(uint8_24ToUInt(explicit_type_conversion<size_t, unsigned>(curve->order()), curveid))));
    }

    return contour_segment_index;
  }

} // namespace gpucast