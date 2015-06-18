/********************************************************************************
*
* Copyright (C) 2013 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : trimdomain_serializer_contour_map_binary.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_CORE_TRIMDOMAIN_SERIALIZER_CONTOUR_MAP_BINARY_HPP
#define GPUCAST_CORE_TRIMDOMAIN_SERIALIZER_CONTOUR_MAP_BINARY_HPP

// header, system
#include <gpucast/math/parametric/domain/partition/monotonic_contour/contour_map_binary.hpp>

// header, project
#include <gpucast/core/gpucast.hpp>

#include <gpucast/core/trimdomain.hpp>
#include <gpucast/core/trimdomain_serializer.hpp>

//#define NO_EXTRA_CURVEINFO_BUFFER

namespace gpucast {

class trimdomain_serializer_contour_map_binary : public trimdomain_serializer
{
  public : // enums/typedefs
 
    typedef trimdomain::value_type                                                      value_type;
    typedef gpucast::math::domain::contour_map_binary<value_type>::contour_segment_ptr  contour_segment_ptr;

  public : // methods

    template <typename float4_type, typename float3_type>
    address_type     serialize  ( trimdomain_ptr const&                                     input_domain, 
                                  std::unordered_map<trimdomain_ptr, address_type>&       referenced_trimdomains,
                                  std::unordered_map<curve_ptr, address_type>&            referenced_curves,
                                  std::unordered_map<contour_segment_ptr, address_type>&  referenced_contour_segments,
                                  std::vector<float4_type>&                                 output_partition,
                                  std::vector<float4_type>&                                 output_contourlist,
                                  std::vector<float4_type>&                                 output_curvelist,
                                  std::vector<float>&                                       output_curvedata,
                                  std::vector<float3_type>&                                 output_pointdata ) const;

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
template <typename float4_type, typename float3_type>
trimdomain_serializer::address_type  
trimdomain_serializer_contour_map_binary::serialize ( trimdomain_ptr const&                                   input_domain, 
                                                      std::unordered_map<trimdomain_ptr, address_type>&       referenced_trimdomains,
                                                      std::unordered_map<curve_ptr, address_type>&            referenced_curves,
                                                      std::unordered_map<contour_segment_ptr, address_type>&  referenced_contour_segments,
                                                      std::vector<float4_type>&                               output_partition,
                                                      std::vector<float4_type>&                               output_contourlist,
                                                      std::vector<float4_type>&                               output_curvelist,
                                                      std::vector<float>&                                     output_curvedata,
                                                      std::vector<float3_type>&                               output_pointdata ) const
{
  typedef gpucast::math::domain::contour_map_binary<beziersurface::curve_point_type::value_type> contour_map_type;
  assert ( output_partition.size() < std::numeric_limits<address_type>::max() );

  // if already in buffer -> return index
  if ( referenced_trimdomains.count( input_domain ) ) 
  {
    return referenced_trimdomains.find ( input_domain )->second;
  }

  address_type partition_index = explicit_type_conversion<std::size_t, address_type>(output_partition.size());
  contour_map_type map;

  // fill loops into contour map
  std::for_each ( input_domain->loops().begin(), input_domain->loops().end(), std::bind ( &contour_map_type::add, &map, std::placeholders::_1 ) );
  map.initialize();
  
  float_type umin = explicit_type_conversion<double, float>(map.bounds().min[point_type::u]);
  float_type umax = explicit_type_conversion<double, float>(map.bounds().max[point_type::u]);
  float_type vmin = explicit_type_conversion<double, float>(map.bounds().min[point_type::v]);
  float_type vmax = explicit_type_conversion<double, float>(map.bounds().max[point_type::v]);

  assert ( map.partition().size() < std::numeric_limits<address_type>::max() ); 
  address_type vintervals = explicit_type_conversion<std::size_t, address_type>(map.partition().size());

  output_partition.resize ( partition_index + 2 + map.partition().size() ); 
  output_partition[partition_index]   = float4_type ( unsigned_bits_as_float(vintervals), 0, 0, 0 );
  output_partition[partition_index+1] = float4_type ( umin, umax, vmin, vmax );
  std::size_t vindex = partition_index + 2;
  
  for ( auto const& vinterval : map.partition() )
  {
    assert ( vinterval.cells.size() < std::numeric_limits<address_type>::max() );

    address_type uid    = explicit_type_conversion<size_t, address_type>(output_partition.size());
    address_type ucells = explicit_type_conversion<size_t, address_type>(vinterval.cells.size());

    output_partition[vindex++] = float4_type(explicit_type_conversion<double, float>(vinterval.interval_v.minimum()),
                                             explicit_type_conversion<double, float>(vinterval.interval_v.maximum()),
                                             unsigned_bits_as_float(uid), 
                                             unsigned_bits_as_float(ucells) );
    output_partition.push_back(float4_type(explicit_type_conversion<double, float>(vinterval.interval_u.minimum()),
                                           explicit_type_conversion<double, float>(vinterval.interval_u.maximum()),
                                           0, 
                                           0 ) );

    for (auto const& cell : vinterval.cells)
    {
      address_type contourlist_id = explicit_type_conversion<std::size_t, unsigned>(output_contourlist.size());
      address_type type_and_contours = uint4ToUInt(0, cell.inside, explicit_type_conversion<size_t, address_type>(cell.overlapping_segments.size()), 0);

      output_partition.push_back(float4_type(explicit_type_conversion<double, float>(cell.interval_u.minimum()),
                                             explicit_type_conversion<double, float>(cell.interval_u.maximum()),
                                             unsigned_bits_as_float(type_and_contours), 
                                             unsigned_bits_as_float(contourlist_id) ) );

      for (auto const& contour_segment : cell.overlapping_segments)
      { 
        address_type ncurves_uincreasing = uint2x16ToUInt(explicit_type_conversion<size_t, address_type>(contour_segment->size()), 
                                                          contour_segment->is_increasing(point_type::u));
        
        address_type curvelist_id = serialize_contour_segment ( contour_segment, referenced_contour_segments, referenced_curves, output_curvelist, output_curvedata, output_pointdata );

        output_contourlist.push_back ( float4_type ( unsigned_bits_as_float ( ncurves_uincreasing ), 
                                                     unsigned_bits_as_float ( curvelist_id ),
                                                     unsigned_bits_as_float ( float2_to_unsigned(contour_segment->bbox().min[point_type::u], contour_segment->bbox().min[point_type::v])), 
                                                     unsigned_bits_as_float ( float2_to_unsigned(contour_segment->bbox().max[point_type::u], contour_segment->bbox().max[point_type::v]))));
      }
    }
  }

  // store domain_ptr to index mapping for later reference
  referenced_trimdomains.insert( std::make_pair (input_domain, partition_index));

  // make sure buffers are still in range of address_type
  if ( output_partition.size() >= std::numeric_limits<address_type>::max() ) 
  {
    throw std::runtime_error("Address exceeds maximum of addressable memory");
  }

  return partition_index;
}

/////////////////////////////////////////////////////////////////////////////
template <typename float4_type, typename float3_type>
trimdomain_serializer::address_type  
trimdomain_serializer_contour_map_binary::serialize_contour_segment  ( contour_segment_ptr const&                              contour_segment,
                                                                       std::unordered_map<contour_segment_ptr, address_type>&  referenced_contour_segments,
                                                                       std::unordered_map<curve_ptr, address_type>&            referenced_curves,
                                                                       std::vector<float4_type>&                               output_curvelist,
                                                                       std::vector<float>&                                     output_curvedata,
                                                                       std::vector<float3_type>&                               output_pointdata ) const
{
  if ( referenced_contour_segments.count ( contour_segment ) )
  {
    return referenced_contour_segments[contour_segment];
  } 

  address_type contour_segment_index = explicit_type_conversion<size_t, address_type>(output_curvelist.size());

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
    output_curvelist.push_back(float4_type(explicit_type_conversion<double, float>(bbox.min[point_type::v]),
                                           explicit_type_conversion<double, float>(bbox.max[point_type::v]),
                                           explicit_type_conversion<double, float>(bbox.min[point_type::u]),
                                           explicit_type_conversion<double, float>(bbox.max[point_type::u])));

    output_curvedata.push_back(float_type(unsigned_bits_as_float(uint8_24ToUInt(explicit_type_conversion<size_t, unsigned>(curve->order()), curveid))));
#endif
  }

  return contour_segment_index;
}

} // namespace gpucast

#endif // GPUCAST_CORE_TRIMDOMAIN_SERIALIZER_DOUBLE_BINARY_HPP
