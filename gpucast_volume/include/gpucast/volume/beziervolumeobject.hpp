/********************************************************************************
*
* Copyright (C) 2007-2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : beziervolumeobject.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_BEZIERVOLUMEOBJECT_HPP
#define GPUCAST_BEZIERVOLUMEOBJECT_HPP

// header, system
#include <map>
#include <memory>

// header, external
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <gpucast/math/vec3.hpp>
#include <gpucast/math/vec4.hpp>

// header, project
#include <gpucast/volume/gpucast.hpp>

#include <gpucast/core/renderer.hpp>

#include <gpucast/volume/beziervolume.hpp>
#include <gpucast/volume/isosurface/fragment/renderinfo.hpp>
#include <gpucast/volume/isosurface/fragment/split_heuristic.hpp>



namespace gpucast {

class nurbsvolumeobject;
class volume_renderer;

///////////////////////////////////////////////////////////////////////////////
class GPUCAST_VOLUME beziervolumeobject : public std::enable_shared_from_this<beziervolumeobject>
{
public : // enums, typedefs

  typedef std::shared_ptr<volume_renderer>         volume_renderer_ptr;
  typedef std::shared_ptr<nurbsvolumeobject>       nurbsvolumeobject_ptr;

  typedef gpucast::beziervolume                      element_type;
  typedef element_type::value_type                   value_type;
  typedef element_type::point_type                   point_type;
  typedef element_type::boundingbox_type             boundingbox_type;
  typedef std::map<std::string, boundingbox_type>    minmaxmap_type;

  typedef std::vector<element_type>                  container_type;
  typedef container_type::iterator                   iterator;
  typedef container_type::const_iterator             const_iterator;

public : // c'tor / d'tor

  beziervolumeobject  ( volume_renderer_ptr const& volume_renderer = volume_renderer_ptr(), unsigned uid = 0);
  ~beziervolumeobject ( );

public : // methods

  void                              swap              ( beziervolumeobject& other );

  void                              add               ( element_type const& );

  void                              parent            ( nurbsvolumeobject_ptr const& n );
  nurbsvolumeobject_ptr const&      parent            ( ) const;

  boundingbox_type                  bbox              () const;
                                                                                               
  const_iterator                    begin             () const;
  const_iterator                    end               () const;
                                                      
  iterator                          begin             ();
  iterator                          end               ();
                                                      
  std::size_t                       size              () const;
  unsigned                          uid               () const;
                                   
  void                              clear             ();

  void                              crop              ( std::string const& attribute_name );

  void                              write             ( std::ostream& os ) const;
  void                              read              ( std::istream& is );

private : // methods

private : // attributes

  std::shared_ptr<nurbsvolumeobject>   _nurbsobject;
  unsigned                               _uid;
  container_type                         _volumes;

};

} // namespace gpucast

#endif // GPUCAST_BEZIERVOLUMEOBJECT_HPP
