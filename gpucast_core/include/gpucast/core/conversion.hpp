/********************************************************************************
*
* Copyright (C) 2007-2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : conversion.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_CORE_CONVERSION_HPP
#define GPUCAST_CORE_CONVERSION_HPP

// header, system

// header, project

namespace gpucast {

  template <typename source_t, typename target_t>
  target_t explicit_type_conversion(source_t const& t) {
    return target_t(t);
  }

}

#endif // GPUCAST_CORE_CONVERSION_HPP
