#ifndef LIBGPUCAST_CLAMP_H
#define LIBGPUCAST_CLAMP_H

///////////////////////////////////////////////////////////////////////////////
template <typename T>
__device__ inline
T clamp ( T const& a, T const& min, T const& max )
{
  return a < min ? min : a > max ? max : a;
}

///////////////////////////////////////////////////////////////////////////////
__device__ inline
float2 clamp ( float2 const& a, float2 const& min, float2 const& max )
{
  return float2_t ( clamp ( a.x, min.x, max.x),
                    clamp ( a.y, min.y, max.y) );

}

///////////////////////////////////////////////////////////////////////////////
__device__ inline
float3 clamp ( float3 const& a, float3 const& min, float3 const& max )
{
  return float3_t ( clamp ( a.x, min.x, max.x),
                    clamp ( a.y, min.y, max.y),
                    clamp ( a.z, min.z, max.z) );

}

///////////////////////////////////////////////////////////////////////////////
__device__ inline
float4 clamp ( float4 const& a, float4 const& min, float4 const& max )
{
  return float4_t ( clamp ( a.x, min.x, max.x),
                    clamp ( a.y, min.y, max.y),
                    clamp ( a.z, min.z, max.z),
                    clamp ( a.w, min.w, max.w) );
}


///////////////////////////////////////////////////////////////////////////////
__device__ inline
int2 clamp ( int2 const& a, int2 const& min, int2 const& max )
{
  return int2_t ( clamp ( a.x, min.x, max.x),
                    clamp ( a.y, min.y, max.y) );

}

///////////////////////////////////////////////////////////////////////////////
__device__ inline
int3 clamp ( int3 const& a, int3 const& min, int3 const& max )
{
  return int3_t ( clamp ( a.x, min.x, max.x),
                    clamp ( a.y, min.y, max.y),
                    clamp ( a.z, min.z, max.z) );

}

///////////////////////////////////////////////////////////////////////////////
__device__ inline
int4 clamp ( int4 const& a, int4 const& min, int4 const& max )
{
  return int4_t ( clamp ( a.x, min.x, max.x),
                  clamp ( a.y, min.y, max.y),
                  clamp ( a.z, min.z, max.z),
                  clamp ( a.w, min.w, max.w) );
}


///////////////////////////////////////////////////////////////////////////////
__device__ inline
uint2 clamp ( uint2 const& a, uint2 const& min, uint2 const& max )
{
  return uint2_t ( clamp ( a.x, min.x, max.x),
                   clamp ( a.y, min.y, max.y) );

}

///////////////////////////////////////////////////////////////////////////////
__device__ inline
uint3 clamp ( uint3 const& a, uint3 const& min, uint3 const& max )
{
  return uint3_t ( clamp ( a.x, min.x, max.x),
                   clamp ( a.y, min.y, max.y),
                   clamp ( a.z, min.z, max.z) );

}

///////////////////////////////////////////////////////////////////////////////
__device__ inline
uint4 clamp ( uint4 const& a, uint4 const& min, uint4 const& max )
{
  return uint4_t ( clamp ( a.x, min.x, max.x),
                   clamp ( a.y, min.y, max.y),
                   clamp ( a.z, min.z, max.z),
                   clamp ( a.w, min.w, max.w) );
}



#endif // LIBGPUCAST_CONSTANTS_H