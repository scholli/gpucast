/********************************************************************************
*
* Copyright (C) 2009-2010 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : opencl_types.h
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef LIBGPUCAST_OPENCL_TYPES_H
#define LIBGPUCAST_OPENCL_TYPES_H

#ifdef CUDA 
  #error "Error: Set GPU types to CUDA or OpenCL"
#else
  #define OPENCL
#endif

// built-in c'tor typedefs
#define float_t   (float)
#define float2_t  (float2)
#define float3_t  (float3)
#define float4_t  (float4)

#define uint_t    (uint)
#define uint2_t   (uint2)
#define uint3_t   (uint3)
#define uint4_t   (uint4)

#define int_t     (int)
#define int2_t    (int2)
#define int3_t    (int3)
#define int4_t    (int4)

#define bool_t    (bool)
#define bool2_t   (bool2)
#define bool3_t   (bool3)
#define bool4_t   (bool4)

// undefine CUDA flags
#define __device__ 

#endif
