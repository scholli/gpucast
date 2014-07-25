#ifndef LIBGPUCAST_COMPUTE_INDEXLISTINDEX_H
#define LIBGPUCAST_COMPUTE_INDEXLISTINDEX_H

#include <math/clamp.h>
#include <math/operator.h>

__device__ 
inline unsigned 
compute_indexlistindex ( unsigned    width,
                         unsigned    height,
                         int2 const& coords,
                         int2 const& tilesize,
                         unsigned    pagesize )
{
  unsigned pixel_index = 0;

  int2 resolution      = int2_t(width, height);
  int2 border          = int2_t(resolution.x%tilesize.x, resolution.y%tilesize.y);
  int2 tile_resolution = int2_t(resolution.x/tilesize.x, resolution.y/tilesize.y) + 
                                clamp ( int2_t ( resolution.x%tilesize.x, resolution.y%tilesize.y ), int2_t(0, 0), int2_t(1, 1) );

  int2 tile_id         = int2_t(coords.x%tilesize.x, coords.y%tilesize.y);
  int2 tile_coords     = int2_t(coords.x/tilesize.x, coords.y/tilesize.y);

  int chunksize         = tilesize.x * tilesize.y * pagesize;

  pixel_index           = tile_coords.y * tile_resolution.x * chunksize + 
                          tile_coords.x * chunksize + 
                          tile_id.y * tilesize.x * pagesize + 
                          tile_id.x * pagesize;

  return pixel_index;
}

#endif

