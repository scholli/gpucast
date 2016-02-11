/********************************************************************************
*
* Copyright (C) 2013 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : trimdomain_serializer_contour_map_kd.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_CORE_TRIMDOMAIN_SERIALIZER_CONTOUR_MAP_KD_HPP
#define GPUCAST_CORE_TRIMDOMAIN_SERIALIZER_CONTOUR_MAP_KD_HPP

// header, system
#include <gpucast/math/parametric/domain/partition/monotonic_contour/contour_map_kd.hpp>

// header, project
#include <gpucast/core/gpucast.hpp>

#include <gpucast/core/trimdomain.hpp>
#include <gpucast/core/trimdomain_serializer.hpp>

//#define NO_EXTRA_CURVEINFO_BUFFER

namespace gpucast {

class trimdomain_serializer_contour_map_kd : public trimdomain_serializer
{
  public : // enums/typedefs
 
    typedef trimdomain::value_type                                value_type;
    typedef gpucast::math::domain::contour_map_kd<value_type>::contour_segment_ptr  contour_segment_ptr;

  public : // methods

    template <typename float4_type, typename float3_type, typename float2_type>
    address_type     serialize(trimdomain_ptr const&                                     input_domain,
      kd_split_strategy                                         kdtree_generation,
      std::unordered_map<trimdomain_ptr, address_type>&         referenced_trimdomains,
      std::unordered_map<curve_ptr, address_type>&              referenced_curves,
      std::unordered_map<contour_segment_ptr, address_type>&    referenced_contour_segments,
      std::vector<float4_type>&                                 output_partition,
      std::vector<float2_type>&                                 output_contourlist,
      std::vector<float4_type>&                                 output_curvelist,
      std::vector<float>&                                       output_curvedata,
      std::vector<float3_type>&                                 output_pointdata,
      std::vector<unsigned char>&                               output_classification_field,
      bool texture_classification_enabled,
      unsigned texture_classification_resolution) const;

    template <typename float4_type, typename float3_type>
    address_type     serialize_contour_segment  ( contour_segment_ptr const&                                contour_segment,
                                                  std::unordered_map<contour_segment_ptr, address_type>&  referenced_contour_segments,
                                                  std::unordered_map<curve_ptr, address_type>&            referenced_curves,
                                                  std::vector<float4_type>&                                 output_curvelist,
                                                  std::vector<float>&                                       output_curvedata,
                                                  std::vector<float3_type>&                                 output_pointdata ) const;
  private : // member

};

/////////////////////////////////////////////////////////////////////////////
template <typename float4_type, typename float3_type, typename float2_type>
trimdomain_serializer::address_type
trimdomain_serializer_contour_map_kd::serialize(trimdomain_ptr const&                                     input_domain,
                                                kd_split_strategy                                         kdtree_generation,
                                                std::unordered_map<trimdomain_ptr, address_type>&         referenced_trimdomains,
                                                std::unordered_map<curve_ptr, address_type>&              referenced_curves,
                                                std::unordered_map<contour_segment_ptr, address_type>&    referenced_contour_segments,
                                                std::vector<float4_type>&                                 output_partition,
                                                std::vector<float2_type>&                                 output_contourlist,
                                                std::vector<float4_type>&                                 output_curvelist,
                                                std::vector<float>&                                       output_curvedata,
                                                std::vector<float3_type>&                                 output_pointdata,
                                                std::vector<unsigned char>&                               output_classification_field,
                                                bool                                                      texture_classification_enabled,
                                                unsigned                                                  texture_classification_resolution) const
{
  typedef gpucast::math::domain::contour_map_kd<typename beziersurface::curve_point_type::value_type> contour_map_type;
  typedef typename contour_map_type::kdtree_type::kdnode_ptr kdnode_ptr;
  assert(output_partition.size() < std::numeric_limits<address_type>::max());

  // if already in buffer -> return index
  if (referenced_trimdomains.count(input_domain)) {
    return referenced_trimdomains.find(input_domain)->second;
  }

  address_type partition_base_index = output_partition.size();
  contour_map_type kdtree(kdtree_generation, texture_classification_enabled, texture_classification_resolution);

  // fill loops into contour map
  std::for_each(input_domain->loops().begin(), input_domain->loops().end(), std::bind(&contour_map_type::add, &kdtree, std::placeholders::_1));
  kdtree.initialize();

  // generate fast pre-classification texture
  address_type classification_id = 0;
  if (texture_classification_enabled) {
    classification_id = trimdomain_serializer::serialize(input_domain, output_classification_field, texture_classification_resolution);
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
  output_partition.resize(partition_base_index + header_size);
  output_partition[partition_base_index] = float4_type(unsigned_bits_as_float(nodes.size() != 0), 
                                                       unsigned_bits_as_float(classification_id),
                                                       unsigned_bits_as_float(kdtree.pre_classification().width()),
                                                       unsigned_bits_as_float(kdtree.pre_classification().height())); // base entry
  // second entry is size of partition
  output_partition[partition_base_index + 1] = float4_type(umin, umax, vmin, vmax); 

  // third entry is size of domain
  output_partition[partition_base_index + 2] = float4_type(input_domain->nurbsdomain().min[point_type::u], 
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
      address_type contourlist_base_index = output_contourlist.size();
      output_partition.push_back(float4_type(unsigned_bits_as_float(node_base_info),
                                             unsigned_bits_as_float(contourlist_base_index),
                                             0,  // spare
                                             0   // spare
                                             ));

      // serialize segments in leaf
      for (auto const& contour_segment : node->overlapping_segments)
      {
        address_type ncurves_uincreasing = uint2x16ToUInt(explicit_type_conversion<size_t, address_type>(contour_segment->size()),
          contour_segment->is_increasing(point_type::u));

        address_type curvelist_id = serialize_contour_segment(contour_segment, referenced_contour_segments, referenced_curves, output_curvelist, output_curvedata, output_pointdata);

        output_contourlist.push_back(float4_type(unsigned_bits_as_float(ncurves_uincreasing),
                                                 unsigned_bits_as_float(curvelist_id),
                                                 unsigned_bits_as_float(float2_to_unsigned(contour_segment->bbox().min[point_type::u], contour_segment->bbox().min[point_type::v])),
                                                 unsigned_bits_as_float(float2_to_unsigned(contour_segment->bbox().max[point_type::u], contour_segment->bbox().max[point_type::v]))));
      }

    }
    else { // node is no leaf -> just add split value and children
      address_type less_id = partition_base_index + header_size + node_offset_map[node->child_less];
      address_type greater_id = partition_base_index + header_size + node_offset_map[node->child_greater];

      output_partition.push_back(float4_type(unsigned_bits_as_float(node_base_info),
                                             explicit_type_conversion<double, float>(node->split_value),
                                             unsigned_bits_as_float(less_id),
                                             unsigned_bits_as_float(greater_id)));
    }
  }

  // store domain_ptr to index mapping for later reference
  referenced_trimdomains.insert(std::make_pair(input_domain, partition_base_index));

  // make sure buffers are still in range of address_type
  if ( output_partition.size() >= std::numeric_limits<address_type>::max() ) 
  {
    throw std::runtime_error("Address exceeds maximum of addressable memory");
  }

  return partition_base_index;
}

/////////////////////////////////////////////////////////////////////////////
template <typename float4_type, typename float3_type>
trimdomain_serializer::address_type  
trimdomain_serializer_contour_map_kd::serialize_contour_segment  ( contour_segment_ptr const&                                contour_segment,
                                                                   std::unordered_map<contour_segment_ptr, address_type>&  referenced_contour_segments,
                                                                   std::unordered_map<curve_ptr, address_type>&            referenced_curves,
                                                                   std::vector<float4_type>&                                 output_curvelist,
                                                                   std::vector<float>&                                       output_curvedata,
                                                                   std::vector<float3_type>&                                 output_pointdata ) const
{
  if ( referenced_contour_segments.count ( contour_segment ) )
  {
    return referenced_contour_segments[contour_segment];
  } 

  address_type contour_segment_index = output_curvelist.size();

  for ( auto i_cptr = contour_segment->begin(); i_cptr != contour_segment->end(); ++i_cptr )
  {
    curve_ptr curve = *i_cptr;


    address_type curveid = 0;
    if ( !curve->is_constant ( point_type::u ) &&
         !curve->is_constant ( point_type::v ) ) // linear curves do not have to be tested
    {
      curveid = trimdomain_serializer::serialize ( curve, referenced_curves, output_pointdata );
    }


    bbox_type bbox; 
    curve->bbox_simple ( bbox );

#ifdef NO_EXTRA_CURVEINFO_BUFFER
    output_curvelist.push_back ( float4_type ( bbox.min[point_type::v], 
                                               bbox.max[point_type::v],
                                               unsigned_bits_as_float ( float2_to_unsigned ( bbox.min[point_type::u], 
                                                                                             bbox.max[point_type::u]) ),
                                               unsigned_bits_as_float ( uint8_24ToUInt(curve->order(), curveid) ) ) );
#else
    output_curvelist.push_back ( float4_type ( bbox.min[point_type::v], 
                                               bbox.max[point_type::v],
                                               bbox.min[point_type::u],
                                               bbox.max[point_type::u] ) );

    output_curvedata.push_back ( float_type ( unsigned_bits_as_float ( uint8_24ToUInt(curve->order(), curveid) ) ) );
#endif
  }

  return contour_segment_index;
}

} // namespace gpucast

#endif // GPUCAST_CORE_TRIMDOMAIN_SERIALIZER_DOUBLE_KD_HPP
