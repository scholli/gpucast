/********************************************************************************
*
* Copyright (C) 2013 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : trimdomain_serializer_double_binary.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_CORE_TRIMDOMAIN_SERIALIZER_DOUBLE_BINARY_HPP
#define GPUCAST_CORE_TRIMDOMAIN_SERIALIZER_DOUBLE_BINARY_HPP

// header, system
#include <gpucast/math/parametric/domain/partition/double_binary/partition.hpp>

// header, project
#include <gpucast/core/gpucast.hpp>
#include <gpucast/core/conversion.hpp>

#include <gpucast/core/trimdomain.hpp>
#include <gpucast/core/trimdomain_serializer.hpp>

namespace gpucast {

class trimdomain_serializer_double_binary : public trimdomain_serializer
{
  public : // enums/typedefs

  public : // methods

    template <typename float4_type, typename float3_type>
    address_type     serialize  ( trimdomain_ptr const&                               input, 
                                  std::unordered_map<trimdomain_ptr, address_type>& referenced_trimdomains,
                                  std::unordered_map<curve_ptr, address_type>&      referenced_curves,
                                  std::vector<float4_type>&                           output_vslabs,
                                  std::vector<float4_type>&                           output_cells,
                                  std::vector<float4_type>&                           output_curvelists,
                                  std::vector<float3_type>&                           output_curves ) const;

    // output_vslabs
    // [ id ]       [ id + 1 ]      ...   [ id + 2 + #vintervals ]
    // #vintervals   umin_total      vmin_0                     vmin_n
    // 0             umin_total      vmax_0                     vmax_n
    // 0             vmin_total      uid_0                      uid_n
    // 0             vmax_total      #uintervals_0              #uintervals_n

    // output_cells
    // [uid]             [ uid + 1 ]      ...   [ uid + 1 + #uintervals ]
    //  umin_total        umin_0                     umin_n
    //  umax_total        umax_0                     umax_n
    //  #uintervals_n     intersects_0               intersects_n
    //  0                 curvelist_id_0             curvelist_id_n

    // output_curvelists
    // [ curvelist_id ]    
    //    # curves         curve_id      curve_p0_u
    //    0                +/-curve_order   +/-curve_p0_v    => + means du/dt > 0; - means du/dt < 0
    //    0                tmin          curve_pn_u
    //    0                tmax          curve_pn_v

    // output_curvedata
    // [ curve_id ]  ... [ curve_id + curve_order ]
    //    wp0x                       wpnx
    //    wp0y                       wpny
    //    w0                         wn



    // "urangeslist"
    // "curvelist"  
    // "curvedata"  

  private : // member

};

/////////////////////////////////////////////////////////////////////////////
template <typename float4_type, typename float3_type>
trimdomain_serializer::address_type  
trimdomain_serializer_double_binary::serialize ( trimdomain_ptr const&                               input_domain, 
                                                 std::unordered_map<trimdomain_ptr, address_type>& referenced_trimdomains,
                                                 std::unordered_map<curve_ptr, address_type>&      referenced_curves,
                                                 std::vector<float4_type>&                           output_vslabs,
                                                 std::vector<float4_type>&                           output_cells,
                                                 std::vector<float4_type>&                           output_curvelists,
                                                 std::vector<float3_type>&                           output_curves ) const
{
  // if already in buffer -> return index
  if ( referenced_trimdomains.count( input_domain ) ) 
  {
    return referenced_trimdomains.find ( input_domain )->second;
  }

  // copy curves to double binary partition
  trimdomain::curve_container curves = input_domain->curves();
  gpucast::math::domain::partition<trimdomain_serializer::point_type>  input_partition ( curves.begin(), curves.end() );

  // initialize partition
  input_partition.initialize();

  // entry point in texture
  assert (output_vslabs.size() < std::numeric_limits<address_type>::max() );
  assert (input_partition.size() < std::numeric_limits<address_type>::max() );

  address_type partition_index = explicit_type_conversion<std::size_t, unsigned>(output_vslabs.size());
  address_type vintervals = explicit_type_conversion<std::size_t, unsigned>(input_partition.size());
 
  float_type vmin = explicit_type_conversion<double, float>(input_partition.get_vertical_interval().minimum());
  float_type vmax = explicit_type_conversion<double, float>(input_partition.get_vertical_interval().maximum());
  float_type umin = explicit_type_conversion<double, float>(input_partition.get_horizontal_interval().minimum());
  float_type umax = explicit_type_conversion<double, float>(input_partition.get_horizontal_interval().maximum());

  output_vslabs.push_back ( float4_type ( unsigned_bits_as_float(vintervals), 0, 0, 0 ) );
  output_vslabs.push_back ( float4_type ( umin, umax, vmin, vmax ) );

  for ( auto v : input_partition )
  {   
    assert ( output_cells.size() < std::numeric_limits<address_type>::max() );
    assert ( v->size() < std::numeric_limits<address_type>::max() );

    output_vslabs.push_back(float4_type(explicit_type_conversion<double, float>(v->get_vertical_interval().minimum()),
                                        explicit_type_conversion<double, float>(v->get_vertical_interval().maximum()),
                                        unsigned_bits_as_float ( address_type (output_cells.size())), 
                                        unsigned_bits_as_float ( address_type (v->size()))));

    output_cells.push_back(float4_type(explicit_type_conversion<double, float>(v->get_horizontal_interval().minimum()),
                                       explicit_type_conversion<double, float>(v->get_horizontal_interval().maximum()),
                                       unsigned_bits_as_float ( address_type (v->size()) ),
                                       0 ) );

    for ( gpucast::math::domain::partition<gpucast::math::point2d>::cell_ptr_type const& c : *v )
    {
      assert ( c->intersections() < std::numeric_limits<address_type>::max());
      assert ( output_curvelists.size() < std::numeric_limits<address_type>::max());
      assert ( c->size() < std::numeric_limits<address_type>::max());
      
      output_cells.push_back(float4_type(explicit_type_conversion<double, float>(c->get_horizontal_interval().minimum()),
                                         explicit_type_conversion<double, float>(c->get_horizontal_interval().maximum()),
                                         unsigned_bits_as_float ( address_type (c->intersections()) ),
                                         unsigned_bits_as_float ( address_type (output_curvelists.size() ) ) ) );

      output_curvelists.push_back ( float4_type ( unsigned_bits_as_float ( address_type (c->size() ) ), 0, 0, 0) );

      for ( gpucast::math::domain::partition<gpucast::math::point2d>::curve_segment_ptr const& cv : *c)
      {
        address_type curve_index = trimdomain_serializer::serialize (cv->curve(), referenced_curves, output_curves );
        //address_type order_and_u_increase = cv->curve()->is_increasing(point_type::u) ? explicit_type_conversion<size_t, int>(cv->curve()->order()) : -explicit_type_conversion<size_t, int>(cv->curve()->order());
        int order_and_u_increase = cv->curve()->is_increasing(point_type::u) ? int(cv->curve()->order()) : -1 * int(cv->curve()->order());

        output_curvelists.push_back ( float4_type( unsigned_bits_as_float ( curve_index ),
                                                   unsigned_bits_as_float ( order_and_u_increase ),
                                                   explicit_type_conversion<double, float>(cv->tmin()),
                                                   explicit_type_conversion<double, float>(cv->tmax())));
      }
    }
  }

  referenced_trimdomains.insert( std::make_pair (input_domain, partition_index));

  // make sure buffers are still in range of address_type
  if ( output_vslabs.size()     >= std::numeric_limits<address_type>::max() ||
       output_cells.size()      >= std::numeric_limits<address_type>::max() ||
       output_curvelists.size() >= std::numeric_limits<address_type>::max() ||
       output_curves.size()     >= std::numeric_limits<address_type>::max() ||
       input_partition.size()   >= std::numeric_limits<address_type>::max() ) 
  {
    throw std::runtime_error("Address exceeds maximum of addressable memory");
  }

  return partition_index;
}


} // namespace gpucast

#endif // GPUCAST_CORE_TRIMDOMAIN_SERIALIZER_DOUBLE_BINARY_HPP
