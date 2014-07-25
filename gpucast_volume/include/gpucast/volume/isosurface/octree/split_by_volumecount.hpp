/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : split_by_volumecount.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_SPLIT_BY_VOLUMECOUNT_HPP
#define GPUCAST_SPLIT_BY_VOLUMECOUNT_HPP

// header, system
#include <cstdlib>

// header, project
#include <gpucast/volume/isosurface/octree/split_criteria.hpp>


namespace gpucast {

class split_by_volumecount : public split_criteria
{
public:

  split_by_volumecount( std::size_t max_depth, 
                        std::size_t max_volumes_per_node );

  virtual bool operator() ( ocnode const& node ) const;

private:

  std::size_t _max_depth;
  std::size_t _max_volumes_per_node;

};

} // namespace gpucast

#endif // GPUCAST_SPLIT_BY_VOLUMECOUNT_HPP
