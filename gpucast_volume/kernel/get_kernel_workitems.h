/********************************************************************************
*
* Copyright (C) 2009-2010 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : get_kernel_workitems.h
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef LIBGPUCAST_GET_KERNEL_WORKITEMS_H
#define LIBGPUCAST_GET_KERNEL_WORKITEMS_H

#include <string>
#include <algorithm>

///////////////////////////////////////////////////////////////////////////////
template <typename function_type>
inline std::size_t get_kernel_workitems ( function_type kernel )
{                           
  cudaFuncAttributes  kernel_attribs;
  cudaError_t err = cudaFuncGetAttributes ( &kernel_attribs, kernel );

  if ( err != cudaSuccess ) 
  { 
    throw std::runtime_error ("cannot retrieve kernel attributes");
  }

  float max_workitems       = std::sqrt(float(kernel_attribs.maxThreadsPerBlock));
  float max_workitems_exp2  = std::log(max_workitems) / std::log(2.0f); 

  std::size_t workitems     = std::size_t ( std::pow ( 2.0f, std::floor(max_workitems_exp2) ) );

  return std::max(workitems, std::size_t(1));
}

#endif