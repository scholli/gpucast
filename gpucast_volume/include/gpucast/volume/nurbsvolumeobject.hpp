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
#ifndef GPUCAST_NURBSVOLUMEOBJECT_HPP
#define GPUCAST_NURBSVOLUMEOBJECT_HPP

#ifdef _MSC_VER
 #pragma warning(disable: 4251)
#endif

// header, system
#include <list>
#include <set>

// header, project
#include <gpucast/volume/gpucast.hpp>

#include <gpucast/volume/nurbsvolume.hpp>

namespace gpucast 
{

class GPUCAST_VOLUME nurbsvolumeobject 
{
public : // enums, typedefs

  typedef gpucast::nurbsvolume                            volume_type;
  typedef volume_type::value_type                         value_type;
  typedef volume_type::point_type                         point_type;
  typedef volume_type::boundingbox_type                   boundingbox_type;

  typedef volume_type::attribute_volume_type::point_type  attribute_type;
  typedef gpucast::math::axis_aligned_boundingbox<attribute_type>   attribute_boundingbox;

  typedef std::vector<volume_type>                        container_type;
  typedef container_type::iterator                        iterator;
  typedef container_type::const_iterator                  const_iterator;

  typedef std::shared_ptr<volume_type>                  volume_ptr;

public : // c'tor / d'tor

  nurbsvolumeobject();
  ~nurbsvolumeobject();

public : // methods

  void                  add            ( nurbsvolume const& b );

  const_iterator        begin          () const;
  const_iterator        end            () const;
  std::size_t           size           () const;

  boundingbox_type      bbox           () const;
  attribute_boundingbox bbox           ( std::string const& attribute ) const;

  std::set<std::string> attribute_list () const;
  void                  normalize_attribute ();

  void                  identify_inner ();

  void                  crop           ( std::string const& attribute );
 
  std::string const&    name           () const;
  void                  name           ( std::string const& );

  void                  write          ( std::ostream& os ) const;
  void                  read           ( std::istream& is );

private : // data members

  container_type        _volumes;
  std::string           _name;

};

} // namespace gpucast

#endif // GPUCAST_NURBSVOLUMEOBJECT_HPP
