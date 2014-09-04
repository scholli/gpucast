/********************************************************************************
*
* Copyright (C) 2007-2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : beziervolume.cpp
*  project    : gpucast
*  description:
*
********************************************************************************/

// header i/f
#include "gpucast/volume/beziervolume.hpp"

// header, system
#include <gpucast/math/vec3.hpp>

// header, project
#include <gpucast/core/convex_hull_impl.hpp>


// explicit instantiation of base class
namespace gpucast {
  namespace math {
    GPUCAST_VOLUME_EXTERN template class GPUCAST_VOLUME beziervolume<point<double, 3> >;
    GPUCAST_VOLUME_EXTERN template class GPUCAST_VOLUME beziervolume<point<float, 3> >;
    GPUCAST_VOLUME_EXTERN template class GPUCAST_VOLUME beziervolume<point<float, 1> >;
  }
}

namespace gpucast {

////////////////////////////////////////////////////////////////////////////////
beziervolume::beziervolume ()
  : base_type       (),
    _data           (),
    _hull           (),
    _is_outer       (),
    _surface_ids    (),
    _neighbor_ids   (),
    _id             () 
{}


////////////////////////////////////////////////////////////////////////////////
beziervolume::beziervolume ( base_type const& v )
  : base_type       ( v ),
    _data           (),
    _hull           (),
    _is_outer       (),
    _surface_ids    (),
    _neighbor_ids   (),
    _id             ()
{}


////////////////////////////////////////////////////////////////////////////////
beziervolume::~beziervolume ()
{}


////////////////////////////////////////////////////////////////////////////////
void 
beziervolume::attach ( std::string const& name, attribute_volume_type const& data )
{
  _data.insert(std::make_pair(name, data));
}


////////////////////////////////////////////////////////////////////////////////
void                    
beziervolume::detach ( std::string const& name )
{
  beziervolume::attribute_volume_map::iterator element = _data.find(name);

  if ( element != _data.end() ) {
    _data.erase(element);
  } 
}


////////////////////////////////////////////////////////////////////////////////
beziervolume::attribute_volume_type const&   
beziervolume::operator[] ( std::string const& name ) const
{
  beziervolume::attribute_volume_map::const_iterator element = _data.find(name);

  if ( element == _data.end() ) {
    throw std::runtime_error("No such data available!" + name);
  }

  return element->second;
}


////////////////////////////////////////////////////////////////////////////////
beziervolume::attribute_volume_map::const_iterator 
beziervolume::data_begin () const
{
  return _data.begin();
}


////////////////////////////////////////////////////////////////////////////////
beziervolume::attribute_volume_map::const_iterator 
beziervolume::data_end () const
{
  return _data.end();
}


////////////////////////////////////////////////////////////////////////////////
beziervolume::attribute_volume_map::iterator 
beziervolume::data_begin ()
{
  return _data.begin();
}


////////////////////////////////////////////////////////////////////////////////
beziervolume::attribute_volume_map::iterator 
beziervolume::data_end ()
{
  return _data.end();
}


////////////////////////////////////////////////////////////////////////////////
std::unordered_set<std::string> 
beziervolume::data_names () const
{
  std::unordered_set<std::string> names;

  for ( attribute_volume_map::value_type const& s : _data )
  {
    names.insert(s.first);
  }

  return names;
}


////////////////////////////////////////////////////////////////////////////////
std::vector<beziervolume::point_type> const&
beziervolume::convexhull ( )
{
  // if not generated already, generate convex hull
  if ( _hull.empty() ) 
  {
    _generate_convexhull();
  }

  return _hull;
}


////////////////////////////////////////////////////////////////////////////////
void
beziervolume::_generate_convexhull ( )
{  
  std::vector<gpucast::math::vec3d> input ( begin(), end() );
  std::vector<gpucast::math::vec3d> output;

  std::vector<point_type> vertices;
  std::vector<int>        indices;

  convex_hull_compute<3, gpucast::math::vec3d>(&(*(input.begin()))[0],
                                      size(),
                                      std::back_inserter(output),
                                      std::back_inserter(indices),
                                      0);

  vertices.resize(output.size());
  std::copy(output.begin(), output.end(), vertices.begin());

  _hull.clear();
  for(int i : indices) {
    _hull.push_back(vertices[i]);
  }
}


////////////////////////////////////////////////////////////////////////////////
/* virtual */ beziervolume::array_type    
beziervolume::split () const
{
  base_type::array_type                                     geometry_ocsplit = base_type::ocsplit();
  std::map<std::string, attribute_volume_type::array_type>  data_ocsplit;

  for ( attribute_volume_map::value_type const& d : _data ) 
  {
    data_ocsplit.insert( std::make_pair(d.first, d.second.ocsplit() ) );
  }

  beziervolume::array_type split;
  for ( std::size_t w = 0; w != 2; ++w ) 
  {
    for ( std::size_t v = 0; v != 2; ++v ) 
    {
      for ( std::size_t u = 0; u != 2; ++u ) 
      {
        // set geometry of sub volume 
        split[u][v][w] = beziervolume ( geometry_ocsplit[u][v][w] );

        // and apply all attached data
        for ( attribute_volume_map::value_type const& d : _data ) {
          split[u][v][w].attach(d.first, data_ocsplit[d.first][u][v][w]);
        }
      }
    }
  }

  return split;
}


////////////////////////////////////////////////////////////////////////////////
void                              
beziervolume::is_outer ( boundary_bool_map const& outer )
{
  _is_outer = outer;
}


////////////////////////////////////////////////////////////////////////////////
beziervolume::boundary_bool_map const&      
beziervolume::is_outer ( ) const
{
  return _is_outer;
}


////////////////////////////////////////////////////////////////////////////////
void                              
beziervolume::is_special ( boundary_bool_map const& special )
{
  _is_special = special;
}


////////////////////////////////////////////////////////////////////////////////
beziervolume::boundary_bool_map const&      
beziervolume::is_special ( ) const
{
  return _is_special;
}


////////////////////////////////////////////////////////////////////////////////
void                              
beziervolume::surface_ids ( boundary_unsigned_map const& ids )
{
  _surface_ids = ids;
}


////////////////////////////////////////////////////////////////////////////////
beziervolume::boundary_unsigned_map const&      
beziervolume::surface_ids ( ) const
{
  return _surface_ids;
}


////////////////////////////////////////////////////////////////////////////////
void                              
beziervolume::neighbor_ids ( boundary_unsigned_map const& ids )
{
  _neighbor_ids = ids;
}


////////////////////////////////////////////////////////////////////////////////
beziervolume::boundary_unsigned_map const&      
beziervolume::neighbor_ids ( ) const
{
  return _neighbor_ids;
}


////////////////////////////////////////////////////////////////////////////////
void                                  
beziervolume::adjacency ( adjacency_map const& uids )
{
  _adjacency_ids = uids;
}


////////////////////////////////////////////////////////////////////////////////
beziervolume::adjacency_map const&                  
beziervolume::adjacency () const
{
  return _adjacency_ids;
}


////////////////////////////////////////////////////////////////////////////////
void 
beziervolume::id ( unsigned id )
{
  _id = id;
}


////////////////////////////////////////////////////////////////////////////////
unsigned                          
beziervolume::id () const
{
  return _id;
}


////////////////////////////////////////////////////////////////////////////////
void                              
beziervolume::crop ( std::string const& attribute_name )
{
  for ( std::map<std::string, attribute_volume_type>::iterator i = _data.begin(); i != _data.end(); ) 
  {
    if ( i->first != attribute_name ) {
      _data.erase(i++);
    } else {
      ++i;
    }
  }
}


////////////////////////////////////////////////////////////////////////////////
void                              
beziervolume::write ( std::ostream& os ) const
{
  base_type::write(os);

  // class member
  os.write ( reinterpret_cast<char const*>(&_id),          sizeof(unsigned));
  std::size_t attached_attributes = _data.size();

  os.write ( reinterpret_cast<char const*>(&attached_attributes), sizeof(std::size_t));

  for ( auto const& p : _data ) 
  {
    std::size_t attribute_name_length = p.first.size();
    os.write ( reinterpret_cast<char const*>(&attribute_name_length), sizeof(std::size_t) );
    os.write ( p.first.c_str(), attribute_name_length );
    p.second.write(os);
  }

  std::size_t hullsize = _hull.size();
  os.write ( reinterpret_cast<char const*>(&hullsize),          sizeof(std::size_t) );

  if ( hullsize > 0 ) {
    os.write ( reinterpret_cast<char const*>(&_hull.front()),     sizeof(point_type) * _hull.size() );
  }

  os.write ( reinterpret_cast<char const*>(&_is_outer[0]),      sizeof(boundary_bool_map) );
  os.write ( reinterpret_cast<char const*>(&_is_special[0]),    sizeof(boundary_bool_map) );
  os.write ( reinterpret_cast<char const*>(&_surface_ids[0]),   sizeof(boundary_unsigned_map) );
  os.write ( reinterpret_cast<char const*>(&_neighbor_ids[0]),  sizeof(boundary_unsigned_map) );
  os.write ( reinterpret_cast<char const*>(&_adjacency_ids[0]), sizeof(adjacency_map) );
}


////////////////////////////////////////////////////////////////////////////////
void                              
beziervolume::read ( std::istream& is )
{
  base_type::read(is);

  // class member
  is.read ( reinterpret_cast<char*>(&_id),        sizeof(unsigned));

  std::size_t attributes;
  is.read ( reinterpret_cast<char*>(&attributes), sizeof(std::size_t));

  for ( unsigned i = 0; i != attributes; ++i ) 
  {
    std::size_t attribute_name_length;
    is.read ( reinterpret_cast<char*>(&attribute_name_length), sizeof(std::size_t) );

    std::string attribute_name;
    attribute_name.resize(attribute_name_length);
    is.read ( reinterpret_cast<char*>(&attribute_name[0]), attribute_name_length );

    beziervolume::attribute_volume_type bv;
    _data.insert( std::make_pair(attribute_name, bv) );
    _data.find(attribute_name)->second.read(is);
  }

  std::size_t hullsize;
  is.read ( reinterpret_cast<char*>(&hullsize),         sizeof(std::size_t) );

  _hull.resize ( hullsize );

  if ( hullsize > 0 ) {
    is.read ( reinterpret_cast<char*>(&_hull.front()),    sizeof(point_type) * hullsize );
  }

  is.read ( reinterpret_cast<char*>(&_is_outer[0]),      sizeof(boundary_bool_map) );
  is.read ( reinterpret_cast<char*>(&_is_special[0]),    sizeof(boundary_bool_map) );
  is.read ( reinterpret_cast<char*>(&_surface_ids[0]),   sizeof(boundary_unsigned_map) );
  is.read ( reinterpret_cast<char*>(&_neighbor_ids[0]),  sizeof(boundary_unsigned_map) );
  is.read ( reinterpret_cast<char*>(&_adjacency_ids[0]), sizeof(adjacency_map) );
}


} // namespace gpucast
