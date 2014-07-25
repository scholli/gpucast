/********************************************************************************
*
* Copyright (C) 2007-2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : hyperspace_adapter.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_HYPERSPACE_ADAPTER_HPP
#define GPUCAST_HYPERSPACE_ADAPTER_HPP

namespace gpucast
{
  template <typename source_t, typename target_t>
  struct hyperspace_adapter_3D_to_4D 
  {
    target_t operator() ( source_t const& s ) const
    {
      typedef typename target_t::value_type value_type;
      return target_t ( value_type(s[0] * s[3]),
                        value_type(s[1] * s[3]),
                        value_type(s[2] * s[3]),
                        value_type(s[3]) );
    }
  };

  template <typename source_t, typename target_t>
  struct hyperspace_adapter_2D_to_3D 
  {
    target_t operator() ( source_t const& s ) const
    {
      typedef typename target_t::value_type value_type;
      return target_t ( value_type(s[0] * s[2]),
                        value_type(s[1] * s[2]),
                        value_type(s[2]) );
    }
  };

  template <typename source_t, typename target_t>
  struct hyperspace_adapter_2D_to_4D 
  {
    target_t operator() ( source_t const& s ) const
    {
      typedef typename target_t::value_type value_type;
      return target_t ( value_type(s[0] * s[2]),
                        value_type(s[1] * s[2]),
                        value_type(s[2]),
                        value_type(0) );
    }
  };
}

#endif // GPUCAST_HPP
