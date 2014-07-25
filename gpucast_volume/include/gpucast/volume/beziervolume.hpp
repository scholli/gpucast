/********************************************************************************
*
* Copyright (C) 2007-2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : beziervolume.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_BEZIERVOLUME_HPP
#define GPUCAST_BEZIERVOLUME_HPP

// header, system
#include <map>
#include <list>
#include <array>
#include <unordered_map>
#include <unordered_set>

// header, external

#include <gpucast/math/parametric/beziervolume.hpp>
#include <gpucast/math/parametric/point.hpp>

// header, project
#include <gpucast/volume/gpucast.hpp>

#include <gpucast/volume/nurbsvolume.hpp>


namespace gpucast {

class GPUCAST_VOLUME beziervolume : public gpucast::math::beziervolume<gpucast::math::point<float,3> >
{
public : // enums, typedefs

  typedef gpucast::math::beziervolume<point_type>                                     base_type;
  typedef nurbsvolume::attribute_type                                       attribute_type;
  typedef gpucast::math::beziervolume<attribute_type>                                 attribute_volume_type;
  typedef std::map<std::string, attribute_volume_type>                      attribute_volume_map;

  typedef std::array<std::array<std::array<beziervolume, 2>, 2>, 2>   array_type;
  typedef std::array<unsigned, 6>                                         boundary_unsigned_map;
  typedef std::array<bool, 6>                                             boundary_bool_map;
  typedef std::array<unsigned, 27>                                        adjacency_map;

public : // c'tor / d'tor

  beziervolume  ( );
  beziervolume  ( base_type const& v );
  ~beziervolume ( );

public : // methods

  void                                  attach                  ( std::string const& name, attribute_volume_type const& data );
  void                                  detach                  ( std::string const& name );

  attribute_volume_type const&          operator[]              ( std::string const& name ) const;

  attribute_volume_map::const_iterator  data_begin              () const;
  attribute_volume_map::iterator        data_begin              ();
  attribute_volume_map::const_iterator  data_end                () const;
  attribute_volume_map::iterator        data_end                ();

  std::unordered_set<std::string>       data_names              () const;

  std::vector<point_type> const&        convexhull              ();
  /* virtual */ array_type              split                   () const;

  void                                  is_outer                ( boundary_bool_map const& ids );
  boundary_bool_map const&              is_outer                () const;

  void                                  is_special              ( boundary_bool_map const& ids );
  boundary_bool_map const&              is_special              () const;

  void                                  surface_ids             ( boundary_unsigned_map const& ids );
  boundary_unsigned_map const&          surface_ids             () const;

  void                                  neighbor_ids            ( boundary_unsigned_map const& ids );
  boundary_unsigned_map const&          neighbor_ids            () const;

  void                                  adjacency               ( adjacency_map const& uids );
  adjacency_map const&                  adjacency               () const;

  void                                  id                      ( unsigned );
  unsigned                              id                      () const;

  void                                  crop                    ( std::string const& );

  void                                  write                   ( std::ostream& os ) const;
  void                                  read                    ( std::istream& is );

private : // auxilliary methods

  void                                  _generate_convexhull    ();

private : // attributes

  attribute_volume_map                  _data;
                                        
  std::vector<point_type>               _hull;      
                                        
  boundary_bool_map                     _is_outer;
  boundary_bool_map                     _is_special;

  boundary_unsigned_map                 _surface_ids;
  boundary_unsigned_map                 _neighbor_ids;
  adjacency_map                         _adjacency_ids;
                                        
  unsigned                              _id;
};


} // namespace gpucast

#endif // GPUCAST_BEZIERVOLUME_HPP
