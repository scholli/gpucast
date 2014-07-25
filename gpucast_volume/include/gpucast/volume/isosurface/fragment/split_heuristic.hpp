/********************************************************************************
*
* Copyright (C) 2007-2012 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : volume_renderer_split_heuristic.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_VOLUME_RENDERER_SPLIT_HEURISTIC_HPP
#define GPUCAST_VOLUME_RENDERER_SPLIT_HEURISTIC_HPP

#include <gpucast/volume/gpucast.hpp>
#include <gpucast/volume/isosurface/fragment/renderbin.hpp>

namespace gpucast {

class GPUCAST_VOLUME split_heuristic
{
public : 
  virtual ~split_heuristic  ();

  virtual bool splitable ( renderbin const&, float& split_position ) const =0;
  virtual void split     ( renderbin const&, renderbin& lhs, renderbin& rhs, float split_position ) const;
};


class GPUCAST_VOLUME binary_split : public split_heuristic
{
public : 
  binary_split ( unsigned max_bins, float split_overhead_percentage );
  /* virtual */ ~binary_split  ();
  
  /* virtual */ bool splitable ( renderbin const&, float& split_position ) const;

private : 

  unsigned  _max;
  float     _overhead;
};


class GPUCAST_VOLUME greedy_split : public split_heuristic
{
public : 
  greedy_split ( unsigned max_bins, float split_overhead_percentage, unsigned split_candidates );
  /* virtual */ ~greedy_split  ();
  
  /* virtual */ bool splitable ( renderbin const&, float& split_position ) const;

private : 

  unsigned  _max;
  float     _overhead;
  unsigned  _candidates;

};

} // namespace gpucast

#endif // GPUCAST_VOLUME_RENDERER_SPLIT_HEURISTIC_HPP