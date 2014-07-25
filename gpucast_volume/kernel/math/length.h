#ifndef LIBGPUCAST_LENGTH_H
#define LIBGPUCAST_LENGTH_H

///////////////////////////////////////////////////////////////////////////////
__device__
inline float length ( float2 const& a )
{
  return sqrt ( a.x * a.x + a.y * a.y );
}

///////////////////////////////////////////////////////////////////////////////
__device__
inline float length ( float3 const& a )
{
  return sqrt ( a.x * a.x + a.y * a.y + a.z * a.z );
}

///////////////////////////////////////////////////////////////////////////////
__device__
inline float length ( float4 const& a )
{
  return sqrt ( a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w );
}

#endif // LIBGPUCAST_LENGTH_H