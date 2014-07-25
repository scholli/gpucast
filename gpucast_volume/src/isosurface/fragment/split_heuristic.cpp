/********************************************************************************
*
* Copyright (C) 2007-2012 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : isosurface/fragment/split_heuristic.cpp
*  project    : gpucast
*  description:
*
********************************************************************************/

// header i/f
#include "gpucast/volume/isosurface/fragment/split_heuristic.hpp"

// header, system

namespace gpucast {

////////////////////////////////////////////////////////////////////////////////
/* virtual */ split_heuristic::~split_heuristic ()
{}


////////////////////////////////////////////////////////////////////////////////
/* virtual */ void split_heuristic::split ( renderbin const& b, renderbin& lhs, renderbin& rhs, float split_position ) const
{
  unsigned overlaps = 0;
  renderchunk::value_type split_candidate  = split_position * ( b.range().minimum() + b.range().maximum() );

  renderchunk::interval_type left_range  ( b.range().minimum(), split_candidate );
  renderchunk::interval_type right_range ( split_candidate, b.range().maximum() );

  lhs.range ( left_range );
  rhs.range ( right_range );

  for ( renderbin::renderchunk_const_iterator c = b.begin(); c != b.end(); ++c )
  {
    if ( lhs.range().overlap ( (*c)->range ) ) {
      lhs.insert ( *c );
    }
    if ( rhs.range().overlap ( (*c)->range ) ) {
      rhs.insert ( *c );
    }
  }
}


////////////////////////////////////////////////////////////////////////////////
binary_split::binary_split ( unsigned max_bins, float split_overhead_percentage )
  : _max      ( max_bins ),
    _overhead ( std::min ( 1.0f, std::max ( 0.0f, split_overhead_percentage ) ) )
{}


////////////////////////////////////////////////////////////////////////////////
/* virtual */ binary_split::~binary_split ()
{} 


////////////////////////////////////////////////////////////////////////////////
/* virtual */ bool 
binary_split::splitable ( renderbin const& b, float& split_candidate_pos ) const
{
  unsigned overlaps = 0;
  split_candidate_pos = 0.5f;
  renderchunk::value_type split_candidate  = split_candidate_pos * ( b.range().minimum() + b.range().maximum() );

  // return false if too few chunks to split
  if  ( b.chunks() < _max ) {
    return false;
  }

  for ( renderbin::renderchunk_const_iterator c = b.begin(); c != b.end(); ++c )
  {
    if ( (*c)->range.in ( split_candidate ) ) {
      ++overlaps;
    }
  }

  float split_overhead = float (overlaps) / b.chunks();

  return split_overhead < _overhead;
}


////////////////////////////////////////////////////////////////////////////////
greedy_split::greedy_split ( unsigned max_bins, float split_overhead_percentage, unsigned split_candidates )
  : _max      ( max_bins ),
    _overhead ( std::min ( 1.0f, std::max ( 0.0f, split_overhead_percentage ) ) ),
    _candidates ( split_candidates )
{}

////////////////////////////////////////////////////////////////////////////////
/* virtual */ greedy_split::~greedy_split  ()
{}
  
////////////////////////////////////////////////////////////////////////////////
/* virtual */ bool greedy_split::splitable ( renderbin const& b, float& split_candidate_pos ) const
{
  // return false if too few chunks to split
  if  ( b.chunks() < _max ) {
    return false;
  }

  // examine different split_positions
  std::map<float, float> ratio_map;
  for ( unsigned i = 0; i != _candidates; ++i )
  {
    unsigned chunk_overlaps = 0;
    unsigned chunk_left     = 0;
    unsigned chunk_right    = 0;

    float tmp_pos = float(std::rand()) / RAND_MAX;
    renderchunk::value_type split_candidate  = tmp_pos * ( b.range().minimum() + b.range().maximum() );
    
    for ( renderbin::renderchunk_const_iterator c = b.begin(); c != b.end(); ++c )
    {
      if ( (*c)->range.in ( split_candidate ) ) {
        ++chunk_overlaps;
      }
      if ( (*c)->range.less ( split_candidate ) ) {
        ++chunk_left;
      }
      if ( (*c)->range.greater ( split_candidate ) ) {
        ++chunk_right;
      }
    }

    float split_overhead = float (chunk_overlaps) / b.chunks();
    float ratio = fabs ( float ( chunk_left ) / chunk_right - 1.0f );

    if ( split_overhead < _overhead && 
         chunk_left > 0 && 
         chunk_right > 0 )
    {
      ratio_map.insert(std::make_pair(ratio, tmp_pos));
    }
  }

  bool split_candidates_available = !ratio_map.empty();
  if ( split_candidates_available )
  {
    split_candidate_pos = ratio_map.begin()->second;
    std::cout << "Split at : " << split_candidate_pos << std::endl;
  }

  return split_candidates_available;
}


} // namespace gpucast

