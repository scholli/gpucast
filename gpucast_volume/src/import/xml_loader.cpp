/********************************************************************************
*
* Copyright (C) 2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : xml_loader.cpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#include "gpucast/volume/import/xml_loader.hpp"

#include <fstream>
#include <string>
#include <vector>
#include <map>

#include <gpucast/volume/import/xml_grammar.hpp>

#include <gpucast/math/parametric/point.hpp>
#include <gpucast/math/parametric/beziervolume.hpp>
#include <gpucast/math/parametric/pointmesh3d.hpp>

namespace gpucast {


/////////////////////////////////////////////////////////////////////////////
class nurbsvolumeobject_adapter
{
private :

  bool        _apply_displacement;
  std::string _displacement_name;

public :

  nurbsvolumeobject_adapter ( std::shared_ptr<nurbsvolumeobject> const& nvol_ptr,
                              bool apply_displacement,
                              std::string const& displacement_name )
    : _object(nvol_ptr),
      _apply_displacement(apply_displacement),
      _displacement_name(displacement_name)
  {}

  void operator()( xml::nurbsvolumeobject const& source )
  {
    // transform file format data to point container
    gpucast::math::pointmesh3d<nurbsvolumeobject::point_type>                         point_data;
    std::map<std::string, gpucast::math::pointmesh3d<nurbsvolumeobject::attribute_type> > other_data;

    for(xml::controlpoint const& cp : source.points)
    {
      nurbsvolumeobject::point_type position;

      for (xml::controlpoint_attribute const& attrib : cp.data)
      {
        if ( attrib.name == "Vertex" )
        {
          assert ( attrib.data.size() >= nurbsvolumeobject::point_type::coordinates );
          position = nurbsvolumeobject::point_type( attrib.data );
        } else {
          if (attrib.name == "Weight" )
          {
            assert ( !attrib.data.empty() );
            position.weight( nurbsvolume::value_type ( attrib.data.front() ) );
          } else {
            nurbsvolumeobject::attribute_type data_point;
            nurbsvolumeobject::value_type     total = 0;
            for ( int i = 0; i != attrib.data.size(); ++i) 
            {
              if ( i < nurbsvolumeobject::attribute_type::coordinates )
              {
                data_point[i]  = nurbsvolume::value_type ( attrib.data[i] );
              }
              total         += nurbsvolume::value_type (attrib.data[i] * attrib.data[i]);
            }

            other_data[attrib.name].push_back ( data_point );

            nurbsvolumeobject::point_type abs_point;
            abs_point[0] = total / attrib.data.size();
            other_data["|" + attrib.name + "|"].push_back(abs_point);
          }
        }
      }

      point_data.push_back(position);
    }

    // nested xml -> order w, v, u -> transpose to u,v,w 
    point_data.width  (source.knots_w.size() - source.degree_w - 1);
    point_data.height (source.knots_v.size() - source.degree_v - 1);
    point_data.depth  (source.knots_u.size() - source.degree_u - 1);

    // inverse order
    point_data.transpose    (2, 1, 0);

    // create a single volume for geometry and displacement
    nurbsvolume nvol;

    // apply data to nurbsvolumeobject
    nvol.set_points ( point_data.begin(), point_data.end() );
    apply_parameter_to_geometry ( source, nvol );

    typedef std::pair<std::string, gpucast::math::pointmesh3d<nurbsvolumeobject::point_type> > meshpair;

    for ( std::map<std::string, gpucast::math::pointmesh3d<nurbsvolumeobject::attribute_type> >::iterator mesh = other_data.begin(); mesh != other_data.end(); ++mesh )
    {
      mesh->second.width  (source.knots_w.size() - source.degree_w - 1);
      mesh->second.height (source.knots_v.size() - source.degree_v - 1);
      mesh->second.depth  (source.knots_u.size() - source.degree_u - 1);

      mesh->second.transpose (2, 1, 0);

      /*if ( _apply_displacement && mesh->first == _displacement_name )
      {
        point_data.displace(mesh->second);
      }*/

      nurbsvolume::attribute_volume_type attachment;
      attachment.set_points( mesh->second.begin(), mesh->second.end() );
      apply_parameter_to_attribute ( source, attachment );

      // create nurbsvolumeobject
      nvol.attach ( mesh->first, attachment, false );
    }

    // add volume to container object
    _object->add(nvol);
  }

  void apply_parameter_to_geometry ( xml::nurbsvolumeobject const& source, nurbsvolume::base_type& target)
  {
    size_t npoints_u = source.knots_u.size() - (source.degree_u + 1);
    size_t npoints_v = source.knots_v.size() - (source.degree_v + 1);
    size_t npoints_w = source.knots_w.size() - (source.degree_w + 1);

    target.degree_u         ( source.degree_u );
    target.degree_v         ( source.degree_v );
    target.degree_w         ( source.degree_w );

    target.numberofpoints_u ( npoints_u );
    target.numberofpoints_v ( npoints_v );
    target.numberofpoints_w ( npoints_w );

    target.knotvector_u     ( source.knots_u.begin(), source.knots_u.end() );
    target.knotvector_v     ( source.knots_v.begin(), source.knots_v.end() );
    target.knotvector_w     ( source.knots_w.begin(), source.knots_w.end() );
  }

  void apply_parameter_to_attribute ( xml::nurbsvolumeobject const& source, nurbsvolume::attribute_volume_type& target)
  {
    size_t npoints_u = source.knots_u.size() - (source.degree_u + 1);
    size_t npoints_v = source.knots_v.size() - (source.degree_v + 1);
    size_t npoints_w = source.knots_w.size() - (source.degree_w + 1);

    target.degree_u         ( source.degree_u );
    target.degree_v         ( source.degree_v );
    target.degree_w         ( source.degree_w );

    target.numberofpoints_u ( npoints_u );
    target.numberofpoints_v ( npoints_v );
    target.numberofpoints_w ( npoints_w );

    target.knotvector_u     ( source.knots_u.begin(), source.knots_u.end() );
    target.knotvector_v     ( source.knots_v.begin(), source.knots_v.end() );
    target.knotvector_w     ( source.knots_w.begin(), source.knots_w.end() );
  }


  std::shared_ptr<nurbsvolumeobject> const& get() const
  {
    return _object;
  }

private :

  std::shared_ptr<nurbsvolumeobject>    _object;

};


/////////////////////////////////////////////////////////////////////////////
xml_loader::xml_loader()
{}

/////////////////////////////////////////////////////////////////////////////
xml_loader::~xml_loader()
{}

/////////////////////////////////////////////////////////////////////////////
bool
xml_loader::load(std::string const& filename, std::shared_ptr<nurbsvolumeobject> const& ptr, bool displace, std::string  attrib )
{
  ptr->name(filename);

  // open file for reading
  std::ifstream ifstr(filename.c_str(), std::ios_base::in);

  // if file ok
  if (ifstr)
  {
    // copy stream to string
    ifstr.unsetf(std::ios::skipws);
    std::string filestr;
    std::copy(std::istream_iterator<char>(ifstr),
              std::istream_iterator<char>(),
              std::back_inserter(filestr));

    ifstr.close();

    // parse string
    return _parse(filestr, ptr, displace, attrib);
  } else {
    std::cerr << "loader::load(): Warning! Could not open file " << filename << std::endl;
    return false;
  }
}

/////////////////////////////////////////////////////////////////////////////
bool
xml_loader::_parse ( std::string const& filestr, 
                     std::shared_ptr<nurbsvolumeobject> const& nvol_ptr,
                     bool displace,
                     std::string const& attrib )
{
  // create grammar and result data structure
  xml::grammar<std::string::const_iterator> xml_grammar;
 
  try 
  {
    std::vector<xml::nurbsvolumeobject> xml_data;
    std::string::const_iterator b = filestr.begin();

    bool parse_success = boost::spirit::qi::phrase_parse(b, filestr.end(), xml_grammar, boost::spirit::ascii::space, xml_data );

    if (parse_success) 
    {
      nurbsvolumeobject_adapter nvadapter(nvol_ptr, displace, attrib);
      std::for_each( xml_data.begin(), xml_data.end(), nvadapter);
      return nvadapter.get() != nullptr;
    } else {
      std::cerr << "loading failed. Unrecognized sequence : \n";
      std::copy(b, filestr.end(), std::ostream_iterator<char>(std::cerr));
      return false;
    }
  } catch (std::exception& e) {
    std::cerr << "loader::parse(): Reading file failed.\n" << std::endl;
    std::cerr << e.what() << std::endl;
  }

  return false;
}

} // namespace gpucast