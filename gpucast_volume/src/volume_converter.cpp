/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : volume_converter.cpp
*  project    : gpucast
*  description:
*
********************************************************************************/

// header i/f
#include "gpucast/volume/volume_converter.hpp"

// header, system
#include <list>
#include <cmath>
#include <functional>

#include <unordered_map>
#include <thread>

#include <boost/optional.hpp>

// header, project
#include <gpucast/math/parametric/pointmesh2d.hpp>
#include <gpucast/math/parametric/beziervolume.hpp>
#include <gpucast/math/parametric/algorithm/converter.hpp>

#include <gpucast/volume/beziervolumeobject.hpp>
#include <gpucast/volume/nurbsvolumeobject.hpp>
#include <gpucast/volume/beziervolume.hpp>
#include <gpucast/volume/nurbsvolume.hpp>
#include <gpucast/volume/uid.hpp>

#include <gpucast/gl/util/timer.hpp>


namespace gpucast {


class volume_converter::impl_t
{
public :
  impl_t()
    : target            ( ),
      source            ( ),
      next              ( ),
      source_access     ( ),
      target_access     ( ),
      nthreads          (16)
  {}

  std::shared_ptr<beziervolumeobject>    target;
  std::shared_ptr<nurbsvolumeobject>     source;
  nurbsvolumeobject::const_iterator        next;
  boost::mutex                             source_access;
  boost::mutex                             target_access;
  std::size_t                              nthreads;
};


////////////////////////////////////////////////////////////////////////////////
volume_converter::volume_converter()
: _impl  (new impl_t)
{}

////////////////////////////////////////////////////////////////////////////////
volume_converter::~volume_converter()
{
  delete _impl;
}


////////////////////////////////////////////////////////////////////////////////
void
volume_converter::convert(std::shared_ptr<nurbsvolumeobject> const& ns, std::shared_ptr<beziervolumeobject> const& bs)
{
  // clear converter "stack" and copy source and target ptr so that convert threads can access simultaneously
  _impl->target = bs;
  _impl->target->clear();
  _impl->target->parent(ns);

  _impl->source = ns;
  _impl->next   = _impl->source->begin();

  // create threadpool and keep threads fetching jobs
  std::list<std::shared_ptr<std::thread> > threadpool;

  for (std::size_t i = 0; i != _impl->nthreads; ++i)
  {
    threadpool.push_back(std::shared_ptr<std::thread>(new std::thread(std::bind(&volume_converter::_fetch_task, this))));
  }

  // wait for all threads to join to finish conversion
  std::for_each(threadpool.begin(), threadpool.end(), std::bind(&std::thread::join, std::placeholders::_1));

  std::cout << "Converting NURBS to Bezier volume:" << std::endl;
  std::cout << "  - NURBS elements  : " << ns->size() << std::endl;
  std::cout << "  - Bezier elements : " << bs->size() << std::endl;

  _identify_neighbors ();
}


////////////////////////////////////////////////////////////////////////////////
void
volume_converter::_convert(nurbsvolumeobject::volume_type const& volume)
{
  // store size of beziervolume array
  unsigned elements_u = volume.knotspans_u();
  unsigned elements_v = volume.knotspans_v();
  unsigned elements_w = volume.knotspans_w();

  // generate unique identifier for volumes and unique and shared surfaces
  unsigned_array3d_t unique_volume_ids  ( boost::extents[elements_u][elements_v][elements_w] ); 
  unsigned_array4d_t unique_surface_ids ( boost::extents[elements_u][elements_v][elements_w][beziervolume::boundary_t::count] );

  _generate_volume_ids  ( elements_u, elements_v, elements_w, unique_volume_ids );
  _generate_surface_ids ( elements_u, elements_v, elements_w, unique_surface_ids );

  // create a nurbsvolume to beziervolume converter
  gpucast::math::converter<nurbsvolume::point_type>     geometry_converter;
  gpucast::math::converter<nurbsvolume::attribute_type> attribute_converter;

  // 1. create container for sub-geometry and sub-volumes
  std::vector<beziervolumeobject::element_type::base_type>  beziervolumes_geometry;
  std::vector<beziervolumeobject::element_type>             beziervolumes;

  // 2. convert geometry first
  std::vector<gpucast::math::beziervolumeindex> indices;
  geometry_converter.convert(volume, std::back_inserter ( beziervolumes_geometry ), indices);

  // 3. create sub-volumes
  std::vector<gpucast::math::beziervolumeindex>::const_iterator index_iterator = indices.begin();
  for (beziervolumeobject::element_type::base_type& geometry : beziervolumes_geometry)
  {
    // copy construct derived class from base class
    gpucast::beziervolume bv (geometry);

    // retrieve indices of beziervolume within nurbsvolume
    unsigned u = index_iterator->u;
    unsigned v = index_iterator->v;
    unsigned w = index_iterator->w;
    ++index_iterator;
    
    // for each side of beziervolume, determine the side's unique id and the neighbor volume's id (0, if non-existant)
    beziervolume::boundary_bool_map     is_outer;
    beziervolume::boundary_bool_map     is_special;
    beziervolume::boundary_unsigned_map surface_ids;
    beziervolume::boundary_unsigned_map neighbor_ids;
    
    for ( unsigned i = beziervolume::boundary_t::umin; i != beziervolume::boundary_t::count; ++i )
    {
      beziervolume::boundary_t surface = static_cast<beziervolume::boundary_t>(i);
      surface_ids[surface]  = _compute_surface_id  ( unique_surface_ids, u, v, w, elements_u, elements_v, elements_w, surface );
      neighbor_ids[surface] = _compute_neighbor_id ( unique_volume_ids,  u, v, w, elements_u, elements_v, elements_w, surface );
      is_outer[surface]     = ( neighbor_ids[surface] == 0 ) && volume.is_outer()[surface];   // outer NURBS is really outer NURBS
      is_special[surface]   = ( neighbor_ids[surface] == 0 ) && (!volume.is_outer()[surface]); // surface is "special" if it is non-outer NURBS, but has no neighbor
    }

    // extract entire adjacency information
    beziervolume::adjacency_map adjacency;
    _extract_adjacency ( unique_volume_ids,  u, v, w, elements_u, elements_v, elements_w, adjacency );

    // apply unique surface identifier and neighbor ids to volume element
    bv.is_outer             ( is_outer );
    bv.is_special           ( is_special );
    bv.surface_ids          ( surface_ids );
    bv.neighbor_ids         ( neighbor_ids );
    bv.id                   ( unique_volume_ids[u][v][w] );
    bv.adjacency            ( adjacency );

    // store volume 
    beziervolumes.push_back ( bv );
  }

  // 4. convert all data attached and attach to beziervolumes
  for ( nurbsvolume::attribute_map_type::const_iterator data_pair = volume.data_begin(); data_pair != volume.data_end(); ++data_pair)
  {
    std::vector<beziervolume::attribute_volume_type> beziervolumes_data;
    attribute_converter.convert ( data_pair->second, std::back_inserter(beziervolumes_data) );

    // make sure conversion of geometry has same size as conversion of attached data
    assert(beziervolumes_data.size() == beziervolumes.size());

    auto bvol_iter = beziervolumes.begin();
    for ( auto const& attribute_volume : beziervolumes_data )
    {
      // attach according sub-data-volume
      bvol_iter->attach( data_pair->first, attribute_volume);
      ++bvol_iter;
    }
  }

  { // exclusive access to target container
    boost::mutex::scoped_lock lck(_impl->target_access);

    // 5. add bezier subvolumes to target object
    for (beziervolumeobject::element_type& bv : beziervolumes)
    {
      _impl->target->add(bv);
    }
  }
}


////////////////////////////////////////////////////////////////////////////////
void
volume_converter::_fetch_task ()
{
  while ( true )
  {
    // try to get a task
    boost::mutex::scoped_lock lck(_impl->source_access);

    if ( _impl->next != _impl->source->end() )
    {
      nurbsvolume const& task = *(_impl->next);
      ++(_impl->next);

      std::cout << "Converting NURBS Volume " << std::endl;

      gpucast::gl::timer t; 
      t.start();

      _convert( task );

      t.stop();
      std::cout << "time elapsed: " << t.result() << std::endl;

    } else {
      break;
    }
  }
}


////////////////////////////////////////////////////////////////////////////////
unsigned 
volume_converter::_compute_surface_id ( unsigned_array4d_t const& ids,
                                        unsigned u, unsigned v, unsigned w, 
                                        unsigned elements_u, unsigned elements_v, unsigned elements_w,
                                        beziervolume::boundary_t surface ) const
{
  // always if possible use a left, bottom or front element
  if ( u > 0 && u < elements_u-1 && surface == beziervolume::boundary_t::umax ) {
    return _compute_surface_id ( ids, u+1, v, w, elements_u, elements_v, elements_w, beziervolume::boundary_t::umin);
  }
  if ( v > 0 && v < elements_v-1 && surface == beziervolume::boundary_t::vmax ) {
    return _compute_surface_id ( ids, u, v+1, w, elements_u, elements_v, elements_w, beziervolume::boundary_t::vmin);
  }
  if ( w > 0 && w < elements_w-1 && surface == beziervolume::boundary_t::wmax ) {
    return _compute_surface_id ( ids, u, v, w+1, elements_u, elements_v, elements_w, beziervolume::boundary_t::wmin);
  }
  return ids[u][v][w][surface];
}


////////////////////////////////////////////////////////////////////////////////
unsigned 
volume_converter::_compute_neighbor_id ( unsigned_array3d_t const& ids,
                                         unsigned u, unsigned v, unsigned w, 
                                         unsigned elements_u, unsigned elements_v, unsigned elements_w,
                                         beziervolume::boundary_t surface ) const
{
  // outer surface -> set neighbor id to 0
  if ( (u == 0 && surface == beziervolume::boundary_t::umin) || 
       (v == 0 && surface == beziervolume::boundary_t::vmin) || 
       (w == 0 && surface == beziervolume::boundary_t::wmin) ||
       (u == elements_u-1 && surface == beziervolume::boundary_t::umax) || 
       (v == elements_v-1 && surface == beziervolume::boundary_t::vmax) || 
       (w == elements_w-1 && surface == beziervolume::boundary_t::wmax) ) 
  {
    return 0;
  }

  // if inner -> compute neighbor id
  switch ( surface )
  {
    case beziervolume::boundary_t::umin : return ids[u-1][v][w]; break;
    case beziervolume::boundary_t::umax : return ids[u+1][v][w]; break;
    case beziervolume::boundary_t::vmin : return ids[u][v-1][w]; break;
    case beziervolume::boundary_t::vmax : return ids[u][v+1][w]; break;
    case beziervolume::boundary_t::wmin : return ids[u][v][w-1]; break;
    case beziervolume::boundary_t::wmax : return ids[u][v][w+1]; break;
  };

  return 0;
}


////////////////////////////////////////////////////////////////////////////////
void                
volume_converter::_identify_neighbors ()
{
  typedef gpucast::math::beziersurface<beziervolume::point_type>    boundary_surface;
  struct volume_boundary 
  {
    boundary_surface         mesh;
    std::size_t              index;
    boundary_surface::bbox_t bbox;
    beziervolume::boundary_t type;
  };
  
  std::vector<volume_boundary> special_boundaries;
 
  std::size_t offset = 0;
  for ( beziervolumeobject::iterator v = _impl->target->begin(); v != _impl->target->end(); ++v, ++offset )
  {
    beziervolume::boundary_bool_map is_special = v->is_special();

    for ( unsigned surface = beziervolume::umin; surface != beziervolume::count; ++surface )
    {
      if ( is_special[surface] ) // if surface is special -> no neighbor, but inner NURBS
      {
        volume_boundary bound;

        bound.type  = beziervolume::boundary_t(surface);
        bound.mesh  = v->slice(bound.type);
        bound.index = offset;
        bound.bbox  = bound.mesh.bbox();

        special_boundaries.push_back(bound);
      }
    }
  }

  std::size_t work_total        = special_boundaries.size();
  std::size_t work_one_percent  = std::max ( work_total / 100, std::size_t(1) );
  std::size_t work_done         = 0;
  
  for ( std::vector<volume_boundary>::const_iterator i = special_boundaries.begin(); i != special_boundaries.end(); ++i )
  {
    ++work_done;
    if ( work_done % work_one_percent == 0 ) {
      std::cout << "\rMatching inner Beziersurfaces ... " << (100.0f * float(work_done)) / work_total << "%";
    }

    for ( std::vector<volume_boundary>::const_iterator j = special_boundaries.begin(); j != special_boundaries.end(); ++j )
    {
      if ( i != j )
      {
        if ( i->bbox.overlap(j->bbox) )
        {
          if ( i->mesh.mesh().equals ( j->mesh.mesh(), beziervolume::value_type(0.00001) ) )
          {
            // get associated volumes
            auto i0 = _impl->target->begin() + i->index;
            auto j0 = _impl->target->begin() + j->index;

            // get volumes' neighbor id's 
            beziervolume::boundary_unsigned_map neighbor_ids_i = i0->neighbor_ids();
            beziervolume::boundary_unsigned_map neighbor_ids_j = j0->neighbor_ids();

            // reset found match
            neighbor_ids_i[i->type] = j0->id();
            neighbor_ids_j[j->type] = i0->id();

            // set new neighbor id array
            i0->neighbor_ids ( neighbor_ids_i );
            j0->neighbor_ids ( neighbor_ids_j );
          }
        }
      }
    }
  }

  std::cout << std::endl << work_done << " special bezier surfaces " << std::endl;

}


////////////////////////////////////////////////////////////////////////////////
void
volume_converter::_generate_volume_ids ( unsigned nelements_u, unsigned nelements_v, unsigned nelements_w, unsigned_array3d_t& ids ) const
{
  for ( unsigned u = 0; u != nelements_u; ++u ) {
    for ( unsigned v = 0; v != nelements_v; ++v ) {
      for ( unsigned w = 0; w != nelements_w; ++w ) {
        ids[u][v][w] = uid::generate("volume");
      }
    }
  }
}


////////////////////////////////////////////////////////////////////////////////
void
volume_converter::_generate_surface_ids ( unsigned nelements_u, unsigned nelements_v, unsigned nelements_w, unsigned_array4d_t& ids ) const
{
  for ( unsigned u = 0; u != nelements_u; ++u ) {
    for ( unsigned v = 0; v != nelements_v; ++v ) {
      for ( unsigned w = 0; w != nelements_w; ++w ) {
        for ( unsigned s = beziervolume::boundary_t::umin; s != beziervolume::boundary_t::count; ++s ) {
          // generate ids only once for inner (equal) surfaces
          if ( (s == beziervolume::boundary_t::umin ||                        // prefer umin, vmin or wmin 
                s == beziervolume::boundary_t::vmin ||  
                s == beziervolume::boundary_t::wmin
               ) || 
               (s == beziervolume::boundary_t::umax && u == nelements_u-1) ||   // only generate umax, vmax, wmax ids when surface is unique ( outer volumes )
               (s == beziervolume::boundary_t::vmax && v == nelements_v-1) ||
               (s == beziervolume::boundary_t::wmax && w == nelements_w-1) 
                )
          {
            ids[u][v][w][s] = uid::generate("surface");
          } 
        }
      }
    }
  }

  // write omitted surface id's
  for ( unsigned u = 0; u != nelements_u; ++u ) {
    for ( unsigned v = 0; v != nelements_v; ++v ) {
      for ( unsigned w = 0; w != nelements_w; ++w ) {
        for ( unsigned s = beziervolume::boundary_t::umin; s != beziervolume::boundary_t::count; ++s ) 
        {
          if ( s == beziervolume::boundary_t::umax && u < nelements_u-1 ) {
            ids[u][v][w][beziervolume::boundary_t::umax] = ids[u+1][v][w][beziervolume::boundary_t::umin];
          }
          if ( s == beziervolume::boundary_t::vmax && v < nelements_v-1 ) {
            ids[u][v][w][beziervolume::boundary_t::vmax] = ids[u][v+1][w][beziervolume::boundary_t::vmin];
          }
          if ( s == beziervolume::boundary_t::wmax && w < nelements_w-1 ) {
            ids[u][v][w][beziervolume::boundary_t::wmax] = ids[u][v][w+1][beziervolume::boundary_t::wmin];
          }
        }
      }
    }
  }
}

void  volume_converter::_extract_adjacency ( unsigned_array3d_t const& ids, 
                                             unsigned u, 
                                             unsigned v, 
                                             unsigned w, 
                                             unsigned nelements_u, 
                                             unsigned nelements_v, 
                                             unsigned nelements_w,
                                             beziervolume::adjacency_map& m ) const
{
  // reset
  m.fill(0);

  unsigned index = 0;
  for ( int du = -1; du <= 1; ++du ) {
    for ( int dv = -1; dv <= 1; ++dv ) {
      for ( int dw = -1; dw <= 1; ++dw ) {
        if ( u + du >= 0 && u + du < nelements_u &&
             v + dv >= 0 && v + dv < nelements_v &&
             w + dw >= 0 && w + dw < nelements_w )
        {
          m[index] = ids[u + du][v + dv][w + dw];
        }
        index++;
      }
    }
  }
}


} // namespace gpucast

