/*******************************************************************************
*
* Copyright (C) 2007-2013 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : xml2.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_XML2_LOADER_HPP
#define GPUCAST_XML2_LOADER_HPP

// header, system
#include <string>
#include <vector>
#include <array>
#include <functional>
#include <cassert>

#include <boost/noncopyable.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <memory>
#include <boost/tokenizer.hpp>
#include <boost/foreach.hpp>

// header, project
#include <gpucast/volume/gpucast.hpp>
#include <gpucast/volume/nurbsvolumeobject.hpp>
#include <gpucast/volume/nurbsvolume.hpp>

namespace gpucast {

///////////////////////////////////////////////////////////////////////////////
class xml2_loader : public boost::noncopyable
{
public : // c'tor / d'tor

  xml2_loader  () {};
  ~xml2_loader () {};

  template <typename container_t>
  container_t tokenize (char const* separator, std::string const& s )
  {
    container_t result;
    boost::char_separator<char> sep(separator);
    boost::tokenizer<boost::char_separator<char>> tokens(s, sep);

    for (const std::string& t : tokens) {
      result.push_back(boost::lexical_cast<typename container_t::value_type>(t));
    }
    return result;
  }

  inline bool load ( std::string const& filename, 
                     std::shared_ptr<nurbsvolumeobject> const& ptr,
                     bool displace = false,
                     std::string = "" )
  {
    using namespace boost::property_tree;

    ptree xml;
    read_xml ( filename, xml );

    for ( auto node = xml.begin(); node != xml.end(); ++node )
    {
      gpucast::nurbsvolume nvol;
      std::vector<gpucast::nurbsvolume::knotvector_type>                                       knotvecs;
      std::vector<unsigned>                                                                    degree;
      std::map<std::string, gpucast::math::pointmesh3d<nurbsvolume::attribute_volume_type::point_type>>  attribs;
      gpucast::math::pointmesh3d<nurbsvolume::point_type>                                                points;
      
      for( ptree::value_type const& nurbsobject : node->second )
      {
        if ( nurbsobject.first == "Dimension" )
        {
          // assumed to be 3 dimensional
        }

        if ( nurbsobject.first == "Degree" )
        {
          degree.push_back(boost::lexical_cast<unsigned>(nurbsobject.second.data()));
        }

        if ( nurbsobject.first == "KnotVector" )
        {
            auto knots = tokenize<nurbsvolume::knotvector_type>(" ", nurbsobject.second.data());
            knotvecs.push_back(knots);
            assert ( nurbsobject.second.get<unsigned>("<xmlattr>.size", true) == knotvecs.back().size() );
        }

        if ( nurbsobject.first == "ControlArray" )
        {
          for ( ptree::value_type const& sequence_w: nurbsobject.second )
          {
            for (ptree::value_type const& sequence_v: sequence_w.second)
            {
              for (ptree::value_type const& sequence_u: sequence_v.second)
              {
                for (ptree::value_type const& cpoint: sequence_u.second)
                {
                  nurbsvolume::point_type point;

                  for (ptree::value_type const& attrib: cpoint.second)
                  {
                    if ( attrib.first == "Vertex" )
                    {
                      std::vector<float> coords = tokenize<std::vector<float>> (" ", attrib.second.data());
                      point = nurbsvolume::point_type (coords[0], coords[1], coords[2]);
                    } else {
                      if ( attrib.first == "Weight" )
                      {
                        point.weight(boost::lexical_cast<float>(attrib.second.data()));
                      } else if ( attrib.first != "<xmlattr>" ) {
                        std::vector<float> coords = tokenize<std::vector<float>> (" ", attrib.second.data());
                        for ( int i = 0; i != coords.size(); ++i )
                        {
                          std::string component_name = attrib.first + boost::lexical_cast<std::string>(i);
                          attribs[component_name].push_back(nurbsvolume::attribute_volume_type::point_type (coords[i], 1.0f));
                        }
                        float sum = std::accumulate(coords.begin(), coords.end(), 0.0f);
                        std::string abs_name = "_abs_" + attrib.first;
                        attribs[abs_name].push_back(nurbsvolume::attribute_volume_type::point_type ( sum / coords.size(), 1.0f));
                      }
                    }
                  }
                  points.push_back(point);
                }
              }
            }
          }
        }
      }

      // nested xml -> order w, v, u -> transpose to u,v,w 
      points.width  (knotvecs[2].size() - degree[2] - 1);
      points.height (knotvecs[1].size() - degree[1] - 1);
      points.depth  (knotvecs[0].size() - degree[0] - 1);

      nvol.set_points(points.begin(), points.end());

      nvol.numberofpoints_u(knotvecs[2].size() - (degree[2] + 1));
      nvol.numberofpoints_v(knotvecs[1].size() - (degree[1] + 1));
      nvol.numberofpoints_w(knotvecs[0].size() - (degree[0] + 1));

      nvol.degree_u(degree[2]);
      nvol.degree_v(degree[1]);
      nvol.degree_w(degree[0]);

      nvol.knotvector_u(knotvecs[2].begin(), knotvecs[2].end());
      nvol.knotvector_v(knotvecs[1].begin(), knotvecs[1].end());
      nvol.knotvector_w(knotvecs[0].begin(), knotvecs[0].end());

      for(auto& attrib : attribs)
      {
        attrib.second.width  (nvol.numberofpoints_w());
        attrib.second.height (nvol.numberofpoints_v());
        attrib.second.depth  (nvol.numberofpoints_u());

        gpucast::nurbsvolume::attribute_volume_type avol;
         
        avol.set_points(attrib.second.begin(), attrib.second.end());

        avol.degree_u(nvol.degree_u());
        avol.degree_v(nvol.degree_v());
        avol.degree_w(nvol.degree_w());

        avol.knotvector_u(nvol.knotvector_u().begin(), nvol.knotvector_u().end());
        avol.knotvector_v(nvol.knotvector_v().begin(), nvol.knotvector_v().end());
        avol.knotvector_w(nvol.knotvector_w().begin(), nvol.knotvector_w().end());

        avol.numberofpoints_u(nvol.numberofpoints_u());
        avol.numberofpoints_v(nvol.numberofpoints_v());
        avol.numberofpoints_w(nvol.numberofpoints_w());

        nvol.attach(attrib.first, avol);
      }

      ptr->add(nvol);
    }
    
    return true;
  }

  
};

}

#endif