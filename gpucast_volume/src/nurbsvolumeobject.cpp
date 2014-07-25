/********************************************************************************
*
* Copyright (C) 2007-2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : nurbsvolumeobject.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/

// header i/f
#include "gpucast/volume/nurbsvolumeobject.hpp"

#include <functional>

namespace gpucast {

////////////////////////////////////////////////////////////////////////////////
nurbsvolumeobject::nurbsvolumeobject()
  : _volumes  (),
    _name     ("unnamed")
{}


////////////////////////////////////////////////////////////////////////////////
nurbsvolumeobject::~nurbsvolumeobject()
{}


////////////////////////////////////////////////////////////////////////////////
void 
nurbsvolumeobject::add( volume_type const& v )
{
  _volumes.push_back(v);
}


////////////////////////////////////////////////////////////////////////////////
nurbsvolumeobject::const_iterator    
nurbsvolumeobject::begin () const
{
  return _volumes.begin();
}


////////////////////////////////////////////////////////////////////////////////
nurbsvolumeobject::const_iterator    
nurbsvolumeobject::end () const
{
  return _volumes.end();
}


////////////////////////////////////////////////////////////////////////////////
std::size_t
nurbsvolumeobject::size () const
{
  return _volumes.size();
}


////////////////////////////////////////////////////////////////////////////////
nurbsvolumeobject::boundingbox_type  
nurbsvolumeobject::bbox () const
{
  if ( _volumes.empty() ) 
  { 
    return boundingbox_type();
  } else {
    boundingbox_type aabb = _volumes.front().bbox();
    std::for_each ( _volumes.begin(), _volumes.end(), [&] ( volume_type const& v ) { aabb.merge(v.bbox()); } );
    return aabb;
  }
}


////////////////////////////////////////////////////////////////////////////////
nurbsvolumeobject::attribute_boundingbox      
nurbsvolumeobject::bbox ( std::string const& attribute ) const
{
  if ( _volumes.empty() ) 
  { 
    return attribute_boundingbox();
  } else {
    attribute_boundingbox aabb = _volumes.front().data(attribute)->second.bbox();
    std::for_each ( _volumes.begin(), _volumes.end(), [&] ( volume_type const& v ) { aabb.merge(v.data(attribute)->second.bbox()); } );
    return aabb;
  }
}


////////////////////////////////////////////////////////////////////////////////
std::set<std::string> 
nurbsvolumeobject::attribute_list () const
{
  std::set<std::string> attributes;
  std::for_each ( _volumes.begin(), 
                  _volumes.end(), 
                  [&] ( volume_type const& v ) 
                  { 
                    for ( nurbsvolumeobject::volume_type::const_map_iterator i = v.data_begin(); i != v.data_end(); ++i) 
                    { 
                      attributes.insert(i->first); 
                    } 
                  }
                );
  return attributes;
}

////////////////////////////////////////////////////////////////////////////////
void                  
nurbsvolumeobject::normalize_attribute ()
{
  std::for_each(_volumes.begin(), _volumes.end(), std::bind(&volume_type::normalize_attribute, std::placeholders::_1));
}


////////////////////////////////////////////////////////////////////////////////
void                  
nurbsvolumeobject::identify_inner ()
{
  std::cout << "Matching outer NURBS ... ";

  int index = 0;
  std::map<int, std::array<gpucast::math::pointmesh2d<point_type>, 6> > outer_surface_map;

  // collect outer_surfaces
  for ( container_type::iterator i = _volumes.begin(); i != _volumes.end(); ++i, ++index)
  {
    gpucast::math::pointmesh3d<nurbsvolume::point_type> mesh3d ( i->points().begin(), i->points().end(), i->numberofpoints_u(), i->numberofpoints_v(), i->numberofpoints_w());
    std::array<gpucast::math::pointmesh2d<point_type>, 6> boundary_surface_points;
    boundary_surface_points[0] = mesh3d.submesh(point_type::u, 0);
    boundary_surface_points[1] = mesh3d.submesh(point_type::u, i->numberofpoints_u()-1);
    boundary_surface_points[2] = mesh3d.submesh(point_type::v, 0);
    boundary_surface_points[3] = mesh3d.submesh(point_type::v, i->numberofpoints_v()-1);
    boundary_surface_points[4] = mesh3d.submesh(point_type::w, 0);
    boundary_surface_points[5] = mesh3d.submesh(point_type::w, i->numberofpoints_w()-1);

    outer_surface_map.insert ( std::make_pair ( index, boundary_surface_points ) );
  }

  std::size_t work_done  = 0;
  std::size_t work_total = outer_surface_map.size();
  std::size_t work_step  = std::max ( std::size_t(work_total/100), std::size_t(1) );
  for ( auto i = outer_surface_map.begin(); i != outer_surface_map.end(); ++i, ++work_done )
  {
    std::array<bool, 6> is_outer = {true, true, true, true, true, true}; // assume all surfaces are outer

    for ( unsigned ik = 0; ik != 6; ++ik ) 
    {
      gpucast::math::pointmesh2d<point_type> const& mesh_i = i->second[ik];

      for ( auto j = outer_surface_map.begin(); j != outer_surface_map.end(); ++j )
      {
        for ( unsigned jk = 0; jk != 6; ++jk ) 
        {
          if ( (j->first != i->first) && // if volumes are not the same
               is_outer[ik] == true ) // if not already identified as an inner surface
          {
            gpucast::math::pointmesh2d<point_type> const& mesh_j = j->second[jk];
            bool meshes_equal = mesh_i.equals ( mesh_j, value_type(0.001) );
            is_outer[ik] &= !meshes_equal;
          }
        }
      }
    }

    _volumes[i->first].is_outer ( is_outer );

    if ( work_total % work_step == 0 )
    {
      std::cout << float(100*work_done)/work_total << " %" << "\r";
    }
  }
  std::cout << std::endl;
}


////////////////////////////////////////////////////////////////////////////////
void                  
nurbsvolumeobject::crop ( std::string const& attribute )
{
  std::for_each ( _volumes.begin(), _volumes.end(), std::bind(&nurbsvolume::crop, std::placeholders::_1, std::cref(attribute)));
}


////////////////////////////////////////////////////////////////////////////////
std::string const&    
nurbsvolumeobject::name () const
{
  return _name;
}


////////////////////////////////////////////////////////////////////////////////
void                 
nurbsvolumeobject::name ( std::string const& name )
{
  _name = name;
}


////////////////////////////////////////////////////////////////////////////////
void                  
nurbsvolumeobject::write ( std::ostream& os ) const
{
  std::size_t volumes = _volumes.size();
  os.write ( reinterpret_cast<char const*> (&volumes), sizeof(std::size_t));
  std::for_each ( _volumes.begin(), _volumes.end(), std::bind(&nurbsvolume::write, std::placeholders::_1, std::ref(os)));

  std::size_t name_length = _name.size();
  os.write ( reinterpret_cast<char const*> (&name_length), sizeof(std::size_t) );
  os.write ( _name.c_str(), name_length );
}

////////////////////////////////////////////////////////////////////////////////
void                  
nurbsvolumeobject::read ( std::istream& is )
{
  std::size_t volumes;
  is.read ( reinterpret_cast<char*>(&volumes), sizeof(std::size_t) );

  _volumes.resize(volumes);
  std::for_each(_volumes.begin(), _volumes.end(), std::bind(&nurbsvolume::read, std::placeholders::_1, std::ref(is)));

  std::size_t name_length;
  is.read ( reinterpret_cast<char*> (&name_length), sizeof(std::size_t) );

  _name.resize ( name_length );
  is.read ( &_name[0], name_length );
}


} // namespace gpucast
