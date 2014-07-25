/********************************************************************************
*
* Copyright (C) 2009-2010 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : cuda_types.h
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef LIBGPUCAST_CUDA_TYPES_H
#define LIBGPUCAST_CUDA_TYPES_H

#ifdef OPENCL 
  #error "Error: Set GPU types to CUDA or OpenCL"
#else
  #define CUDA
#endif

// built-in c'tor typedefs
#define float_t   float
#define float2_t  make_float2
#define float3_t  make_float3
#define float4_t  make_float4

#define uint_t    uint
#define uint2_t   make_uint2
#define uint3_t   make_uint3
#define uint4_t   make_uint4

#define int_t     int
#define int2_t    make_int2
#define int3_t    make_int3
#define int4_t    make_int4

// define CUDA macros and undefine OpenCL flags
#define __kernel __global__
#define __global
#define __local
#define __read_only
#define __write_only

typedef surface<void, cudaSurfaceType2D>                            image2d_t;

#endif