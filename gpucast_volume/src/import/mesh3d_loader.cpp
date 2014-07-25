/********************************************************************************
*
* Copyright (C) 2012 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : xml_loader.cpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#include "gpucast/volume/import/mesh3d_loader.hpp"

#include <fstream>
#include <string>
#include <vector>
#include <array>
#include <sstream>
#include <map>

#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/tokenizer.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string/erase.hpp>
#include <boost/algorithm/string/trim.hpp>

#include <gpucast/math/parametric/point.hpp>
#include <gpucast/math/parametric/beziervolume.hpp>
#include <gpucast/math/parametric/pointmesh3d.hpp>

namespace gpucast {

//#define APPLY_DISPLACEMENT_TO_GEOMETRY

/////////////////////////////////////////////////////////////////////////////
mesh3d_loader::mesh3d_loader()
  : _attribute_scale ( 1.0f )
{}

/////////////////////////////////////////////////////////////////////////////
mesh3d_loader::~mesh3d_loader()
{}

/////////////////////////////////////////////////////////////////////////////
bool
mesh3d_loader::load(std::string const& filename, std::shared_ptr<nurbsvolumeobject> const& ptr,
                        bool displace, std::string name )
{
  if ( boost::filesystem::exists ( filename ) )
  {
    // parse string
    ptr->name(filename);
    return _parse(filename, ptr, displace, name);
  } else {
    std::cerr << "loader::load(): Warning! Could not open file " << filename << std::endl;
    return false;
  }
}


/////////////////////////////////////////////////////////////////////////////
bool
mesh3d_loader::load(std::string const& filename, std::shared_ptr<nurbssurfaceobject> const& ptr)
{
  if ( boost::filesystem::exists ( filename ) )
  {
    // parse string
    return _parse (filename, ptr);
  } else {
    std::cerr << "loader::load(): Warning! Could not open file " << filename << std::endl;
    return false;
  }
}


/////////////////////////////////////////////////////////////////////////////
bool
mesh3d_loader::_parse ( std::string const& file, std::shared_ptr<nurbsvolumeobject> const& nvol_ptr,
                        bool displace, std::string const& name )
{
  try 
  {
    // parse xml 
    boost::property_tree::ptree structure;
    boost::property_tree::read_xml ( file, structure ); 

    // retrieve base directory
    std::string directory = boost::filesystem::path(file).branch_path().string();

    // get header info
    std::string str_scalef     = structure.get<std::string>("meshinfo.head.scale");
    _geometry_basename         = structure.get<std::string>("meshinfo.head.geometry_basename");
    _solution_basename         = structure.get<std::string>("meshinfo.head.solution_basename");
    std::string str_attributes = structure.get<std::string>("meshinfo.head.attributes");
    std::string str_patchinfo  = structure.get<std::string>("meshinfo.patchinfo");

    // trim unnecessary whitespaces
    boost::trim(str_scalef);
    boost::trim(_geometry_basename);
    boost::trim(_solution_basename);

    // add working path to filenames 
    _geometry_basename = directory + "/" + _geometry_basename;
    _solution_basename = directory + "/" + _solution_basename;

    // get model scale
    try {
      //_geometry_scale = boost::lexical_cast<float>(str_scalef);
      _geometry_scale = float(atof(str_scalef.c_str()));
    } catch ( std::exception& e ) {
      std::cerr << "mesh3d_loader::_parse() : Failed to convert model scale." << e.what() << std::endl;
    }
    
    // get attribute names 
    boost::char_separator<char> whitespace_tokens (" \t\n");
    boost::tokenizer<boost::char_separator<char>> attrib_tokenizer ( str_attributes, whitespace_tokens );
    for ( boost::tokenizer<boost::char_separator<char>>::const_iterator i = attrib_tokenizer.begin(); i != attrib_tokenizer.end(); ++i )
    {
      std::string attribute_name       = i.current_token(); ++i;
      unsigned    attribute_dimensions = boost::lexical_cast<unsigned>(i.current_token());
      _solution_attributes.insert ( std::make_pair ( attribute_name, attribute_dimensions ) );
    }

    // extract file names
    boost::char_separator<char> line_separator ("\n");
    boost::tokenizer<boost::char_separator<char>> mappings_tokenizer ( str_patchinfo, line_separator );
    for ( std::string const& line : mappings_tokenizer )
    {
      if ( line.size() > 5 )
      {
        std::stringstream sstr ( line );
        unsigned patch;
        std::array<unsigned, 4> dim;
        std::string delim;
        sstr >> dim[0] >> dim[1] >> dim[2] >> dim[3] >> delim >> patch;

        _geometry_structure[0].insert(dim[0]);
        _geometry_structure[1].insert(dim[1]);
        _geometry_structure[2].insert(dim[2]);
        _geometry_structure[3].insert(dim[3]);

        assert ( delim == "patch:" );

        std::string geometry_filename = (boost::format(_geometry_basename) % dim[0] % dim[1] % dim[2] % dim[3]).str();
        std::string solution_filename = (boost::format(_solution_basename) % patch).str();

        _geometry_solution_mapping.insert  ( std::make_pair ( geometry_filename, solution_filename));
        _geometry_structure_mapping.insert ( std::make_pair ( geometry_filename, dim ) );
      }
    }

    return _load_volumes ( nvol_ptr, displace, name );
  } catch (std::exception& e) {
    std::cerr << "structure::parse(): Reading file failed.\n" << std::endl;
    std::cerr << e.what() << std::endl;
    return false;
  }
}



/////////////////////////////////////////////////////////////////////////////
bool
mesh3d_loader::_parse ( std::string const& file, std::shared_ptr<nurbssurfaceobject> const& nsurf_ptr )
{
  try 
  {
    // parse xml 
    boost::property_tree::ptree structure;
    boost::property_tree::read_xml ( file, structure ); 

    // retrieve base directory
    std::string directory = boost::filesystem::path(file).branch_path().string();

    // get header info
    std::string str_scalef     = structure.get<std::string>("meshinfo.head.scale");
    _geometry_basename         = structure.get<std::string>("meshinfo.head.geometry_basename");
    _solution_basename         = structure.get<std::string>("meshinfo.head.solution_basename");
    std::string str_attributes = structure.get<std::string>("meshinfo.head.attributes");
    std::string str_patchinfo  = structure.get<std::string>("meshinfo.patchinfo");

    // trim unnecessary whitespaces
    boost::trim(str_scalef);
    boost::trim(_geometry_basename);
    boost::trim(_solution_basename);

    // add working path to filenames 
    _geometry_basename = directory + "/" + _geometry_basename;
    _solution_basename = directory + "/" + _solution_basename;

    // get model scale
    try {
      //_geometry_scale = boost::lexical_cast<float>(str_scalef);
      _geometry_scale = float(atof(str_scalef.c_str()));
    } catch ( std::exception& e ) {
      std::cerr << "mesh3d_loader::_parse() : Failed to convert model scale." << e.what() << std::endl;
    }
    
    // get attribute names 
    boost::char_separator<char> whitespace_tokens (" \t\n");
    boost::tokenizer<boost::char_separator<char>> attrib_tokenizer ( str_attributes, whitespace_tokens );
    for ( boost::tokenizer<boost::char_separator<char>>::const_iterator i = attrib_tokenizer.begin(); i != attrib_tokenizer.end(); ++i )
    {
      std::string attribute_name       = i.current_token(); ++i;
      unsigned    attribute_dimensions = boost::lexical_cast<unsigned>(i.current_token());
      _solution_attributes.insert ( std::make_pair ( attribute_name, attribute_dimensions ) );
    }

    // extract file names
    boost::char_separator<char> line_separator ("\n");
    boost::tokenizer<boost::char_separator<char>> mappings_tokenizer ( str_patchinfo, line_separator );
    for ( std::string const& line : mappings_tokenizer )
    {
      if ( line.size() >= 1 )
      {
        std::stringstream sstr ( line );
        unsigned count;
        sstr >> count;

        std::string geometry_filename = (boost::format(_geometry_basename) % count).str();
        std::string solution_filename = (boost::format(_solution_basename) % count).str();

        _geometry_solution_mapping.insert ( std::make_pair ( geometry_filename, solution_filename ) );
      }
    }

    return _load_surfaces ( nsurf_ptr );
  } catch (std::exception& e) {
    std::cerr << "structure::parse(): Reading file failed.\n" << std::endl;
    std::cerr << e.what() << std::endl;
    return false;
  }
}



/////////////////////////////////////////////////////////////////////////////
bool
mesh3d_loader::_load_volumes ( std::shared_ptr<nurbsvolumeobject> const& nvol_ptr,
                               bool apply_displacement,
                               std::string const& displacement_name )
{
  for ( auto const& mapped_solution : _geometry_solution_mapping )
  {
    if ( boost::filesystem::exists ( mapped_solution.first ) &&
         boost::filesystem::exists ( mapped_solution.second ) )
    {
      std::cout << "Reading : " << mapped_solution.first << " : " << mapped_solution.second << std::endl;

      // create target volume 
      nurbsvolume target_volume;

      std::ifstream geometry_ifstr ( mapped_solution.first, std::ios_base::in );

      if ( geometry_ifstr.good() )
      {
        std::istreambuf_iterator<char> fbegin ( geometry_ifstr );
        std::istreambuf_iterator<char> fend;

        boost::char_separator<char> separator (" \t\n");
        typedef boost::tokenizer<boost::char_separator<char>,std::istreambuf_iterator<char>> tokenizer_t;
        tokenizer_t input_str ( fbegin, fend, separator );

        // read header
        std::array<unsigned, 7>           header; // dim, du, dv, dw, cpu, cpv, cpw
        std::array<unsigned, 7>::iterator header_iter = header.begin();
        tokenizer_t::const_iterator       input_iter = input_str.begin();
        for ( int i = 0; i != header.size(); ++i, ++input_iter )
        {
          header_iter[i] = boost::lexical_cast<unsigned>(input_iter.current_token());
        }

        unsigned deg_u    = header[1];
        unsigned deg_v    = header[2];
        unsigned deg_w    = header[3];

        unsigned cp_u     = header[4];
        unsigned cp_v     = header[5];
        unsigned cp_w     = header[6];

        unsigned npoints = cp_u * cp_v * cp_w;

        unsigned nknots_u = cp_u + deg_u + 1;
        unsigned nknots_v = cp_v + deg_v + 1;
        unsigned nknots_w = cp_w + deg_w + 1;

        std::vector<float> knots_u;
        std::vector<float> knots_v;
        std::vector<float> knots_w;

        for ( int i = 0; i != nknots_u; ++i ) {
          knots_u.push_back ( boost::lexical_cast<float>(input_iter.current_token())); ++input_iter;
        }
        for ( int i = 0; i != nknots_v; ++i ) {
          knots_v.push_back ( boost::lexical_cast<float>(input_iter.current_token())); ++input_iter;
        }
        for ( int i = 0; i != nknots_w; ++i ) {
          knots_w.push_back ( boost::lexical_cast<float>(input_iter.current_token())); ++input_iter;
        }

        // open solution in parallel
        std::ifstream solution_ifstr ( mapped_solution.second, std::ios_base::in );
        std::istreambuf_iterator<char> sbegin ( solution_ifstr );
        tokenizer_t solution_str ( sbegin, fend, separator );
        tokenizer_t::const_iterator solution_iter = solution_str.begin();

        typedef nurbsvolumeobject::point_type::value_type   value_type;
        gpucast::math::pointmesh3d<nurbsvolume::point_type>           point_data;

        // first: parse in control points
        for ( unsigned i = 0; i != npoints; ++i ) 
        {
          value_type x      = boost::lexical_cast<float>(input_iter.current_token()); ++input_iter;
          value_type y      = boost::lexical_cast<float>(input_iter.current_token()); ++input_iter;
          value_type z      = boost::lexical_cast<float>(input_iter.current_token()); ++input_iter;
          value_type weight = boost::lexical_cast<float>(input_iter.current_token()); ++input_iter;

          nurbsvolumeobject::point_type point(_geometry_scale * x, _geometry_scale * y, _geometry_scale * z, weight);

          point_data.push_back ( point );
        }

        // apply geometric data to target volume 
        target_volume.set_points(point_data.begin(), point_data.end());

        target_volume.degree_u(deg_u);
        target_volume.degree_v(deg_v);
        target_volume.degree_w(deg_w);
        
        target_volume.numberofpoints_u(cp_u);
        target_volume.numberofpoints_v(cp_v);
        target_volume.numberofpoints_w(cp_w);

        target_volume.knotvector_u ( knots_u.begin(), knots_u.end() );
        target_volume.knotvector_v ( knots_v.begin(), knots_v.end() );
        target_volume.knotvector_w ( knots_w.begin(), knots_w.end() );

#if 1
        std::array<unsigned, 4> const& local_dims   = _geometry_structure_mapping[mapped_solution.first];

        std::array<bool, 6> is_outer = { true,
                                           true,
                                           true,
                                           true,
                                           true,
                                           true };

        target_volume.is_outer ( is_outer );
#endif

        //   - read data for all attributes
        for ( auto const& attrib : _solution_attributes )
        {
          // make sure point can handle solution dimensions
          assert ( nurbsvolumeobject::point_type::coordinates + 1 >= attrib.second );

          gpucast::math::pointmesh3d<nurbsvolume::point_type> solution_mesh;

          for ( unsigned i = 0; i != npoints; ++i ) 
          {
            nurbsvolumeobject::point_type solution_point;
            for ( unsigned j = 0; j != attrib.second; ++j )
            {            
              try {
                if ( !solution_iter.current_token().empty()) {
                  std::string value = solution_iter.current_token();
                  boost::trim(value);
                  solution_point[j] = boost::lexical_cast<value_type>(value);
                }
              } catch ( std::exception& e ) 
              {
                std::cerr << "ERROR: mesh3d_loader::_load_volumes() : " << e.what() << std::endl;
                std::cerr << "***" << solution_iter.current_token() << "***" << std::endl;
              }
              ++solution_iter;
            }
            solution_mesh.push_back(solution_point); 
          }

          // apply displacement to geometry if desired
          if ( apply_displacement && attrib.first == displacement_name ) 
          {
            nurbsvolume::mesh_type displaced_mesh = target_volume.points();
            for ( nurbsvolume::mesh_type::iterator i = displaced_mesh.begin(), j = solution_mesh.begin(); i != displaced_mesh.end(); ++i, ++j )
            {
              value_type weight = i->weight();
              (*i) += (*j);
              i->weight(weight);
            }
            target_volume.set_points(displaced_mesh.begin(), displaced_mesh.end());
          }

          // apply mesh to volume
          nurbsvolume::attribute_volume_type solution;

          solution.degree_u(deg_u);
          solution.degree_v(deg_v);
          solution.degree_w(deg_w);
          
          solution.numberofpoints_u(cp_u);
          solution.numberofpoints_v(cp_v);
          solution.numberofpoints_w(cp_w);

          solution.knotvector_u ( knots_u.begin(), knots_u.end() );
          solution.knotvector_v ( knots_v.begin(), knots_v.end() );
          solution.knotvector_w ( knots_w.begin(), knots_w.end() );

          gpucast::math::pointmesh3d<nurbsvolumeobject::attribute_type> attribute_mesh;
          attribute_mesh.resize ( solution_mesh.size() );
          std::transform ( solution_mesh.begin(), solution_mesh.end(), attribute_mesh.begin(), [] ( nurbsvolume::point_type const a ) { return nurbsvolumeobject::attribute_type(a.abs(), 1.0); } );
          solution.set_points ( attribute_mesh.begin(), attribute_mesh.end() );

          target_volume.attach ( "abs_" + attrib.first, solution );
          
          for ( int c = 0; c != attrib.second; ++c )
          {
            // apply mesh to volume
            nurbsvolume::attribute_volume_type solution;

            solution.degree_u(deg_u);
            solution.degree_v(deg_v);
            solution.degree_w(deg_w);
          
            solution.numberofpoints_u(cp_u);
            solution.numberofpoints_v(cp_v);
            solution.numberofpoints_w(cp_w);

            solution.knotvector_u ( knots_u.begin(), knots_u.end() );
            solution.knotvector_v ( knots_v.begin(), knots_v.end() );
            solution.knotvector_w ( knots_w.begin(), knots_w.end() );

            gpucast::math::pointmesh3d<nurbsvolumeobject::attribute_type> attribute_mesh;
            attribute_mesh.resize ( solution_mesh.size() );
            std::transform ( solution_mesh.begin(), solution_mesh.end(), attribute_mesh.begin(), [&] ( nurbsvolume::point_type const a ) { return nurbsvolumeobject::attribute_type(a[c], 1.0); } );
            solution.set_points ( attribute_mesh.begin(), attribute_mesh.end() );

            target_volume.attach ( attrib.first + "." + boost::lexical_cast<std::string>(c), solution );
          }
        }
        solution_ifstr.close();
      }
      geometry_ifstr.close();

      nvol_ptr->add(target_volume);

      std::cout << "Adding NURBS-Volume : " << target_volume.numberofpoints_u() << ", " << target_volume.numberofpoints_v() << ", " << target_volume.numberofpoints_w() << std::endl;
      std::cout << "        - knotspans : " << target_volume.knotspans_u() << ", " << target_volume.knotspans_v() << ", " << target_volume.knotspans_w() << std::endl;
    }
  }

  nvol_ptr->identify_inner();

  return true;
}



/////////////////////////////////////////////////////////////////////////////
bool
mesh3d_loader::_load_surfaces ( std::shared_ptr<nurbssurfaceobject> const& nsurf_ptr )
{
  for ( auto const& mapped_solution : _geometry_solution_mapping )
  {
    if ( boost::filesystem::exists ( mapped_solution.first ) &&
         boost::filesystem::exists ( mapped_solution.second ) )
    {
      std::cout << "Reading : " << mapped_solution.first << " : " << mapped_solution.second << std::endl;

      // create target volume 
      nurbssurface target_surface;

      std::ifstream geometry_ifstr ( mapped_solution.first, std::ios_base::in );

      if ( geometry_ifstr.good() )
      {
        std::istreambuf_iterator<char> fbegin ( geometry_ifstr );
        std::istreambuf_iterator<char> fend;

        boost::char_separator<char> separator (" \t\n");
        typedef boost::tokenizer<boost::char_separator<char>,std::istreambuf_iterator<char>> tokenizer_t;
        tokenizer_t input_str ( fbegin, fend, separator );

        // read header
        std::array<unsigned, 5>           header; // dim, du, dv, cpu, cpv
        std::array<unsigned, 5>::iterator header_iter = header.begin();
        tokenizer_t::const_iterator       input_iter = input_str.begin();
        for ( int i = 0; i != header.size(); ++i, ++input_iter )
        {
          header_iter[i] = boost::lexical_cast<unsigned>(input_iter.current_token());
        }

        unsigned deg_u    = header[1];
        unsigned deg_v    = header[2];

        unsigned cp_u     = header[3];
        unsigned cp_v     = header[4];

        unsigned npoints = cp_u * cp_v;

        unsigned nknots_u = cp_u + deg_u + 1;
        unsigned nknots_v = cp_v + deg_v + 1;

        std::vector<float> knots_u;
        std::vector<float> knots_v;

        for ( int i = 0; i != nknots_u; ++i ) {
          knots_u.push_back ( boost::lexical_cast<float>(input_iter.current_token())); ++input_iter;
        }
        for ( int i = 0; i != nknots_v; ++i ) {
          knots_v.push_back ( boost::lexical_cast<float>(input_iter.current_token())); ++input_iter;
        }

        typedef nurbsvolumeobject::point_type::value_type   value_type;
        gpucast::math::pointmesh3d<nurbsvolume::point_type>           point_data;

        // first: parse in control points
        for ( unsigned i = 0; i != npoints; ++i ) 
        {
          value_type x      = boost::lexical_cast<float>(input_iter.current_token()); ++input_iter;
          value_type y      = boost::lexical_cast<float>(input_iter.current_token()); ++input_iter;
          value_type z      = boost::lexical_cast<float>(input_iter.current_token()); ++input_iter;
          value_type weight = boost::lexical_cast<float>(input_iter.current_token()); ++input_iter;

          nurbsvolumeobject::point_type point(_geometry_scale * x, _geometry_scale * y, _geometry_scale * z, weight);

          point_data.push_back ( point );
        }

        // apply geometric data to target volume 
        target_surface.set_points(point_data.begin(), point_data.end());

        target_surface.degree_u(deg_u);
        target_surface.degree_v(deg_v);

        target_surface.numberofpoints_u(cp_u);
        target_surface.numberofpoints_v(cp_v);

        target_surface.knotvector_u ( knots_u.begin(), knots_u.end() );
        target_surface.knotvector_v ( knots_v.begin(), knots_v.end() );

#ifdef  APPLY_DISPLACEMENT_TO_GEOMETRY

        // open solution in parallel
        std::ifstream solution_ifstr ( mapped_solution.second, std::ios_base::in );
        std::istreambuf_iterator<char> sbegin ( solution_ifstr );
        tokenizer_t solution_str ( sbegin, fend, separator );
        tokenizer_t::const_iterator solution_iter = solution_str.begin();

        //   - read data for all attributes
        for ( auto const& attrib : _solution_attributes )
        { 
          if ( attrib.first == "displacement" )
          {
            // make sure point can handle solution dimensions
            assert ( nurbsvolumeobject::point_type::coordinates + 1 >= attrib.second );

            gpucast::nurbssurface::container_type displacement_mesh;

            for ( unsigned i = 0; i != npoints; ++i ) 
            {
              nurbsvolumeobject::point_type solution_point;
              for ( unsigned j = 0; j != attrib.second; ++j )
              {            
                try {
                  if ( !solution_iter.current_token().empty()) {
                    std::string value = solution_iter.current_token();
                    boost::trim(value);
                    solution_point[j] = boost::lexical_cast<value_type>(value);
                  }
                } catch ( std::exception& e ) 
                {
                  std::cerr << "ERROR: mesh3d_loader::_load_volumes() : " << e.what() << std::endl;
                  std::cerr << "***" << solution_iter.current_token() << "***" << std::endl;
                }
                ++solution_iter;
              }
              displacement_mesh.push_back(solution_point); 
            }

            gpucast::nurbssurface::container_type displaced_mesh = target_surface.points();
            for ( nurbssurface::container_type::iterator i = displaced_mesh.begin(), j = displacement_mesh.begin(); i != displaced_mesh.end(); ++i, ++j )
            {
              nurbssurface::value_type weight = i->weight();
              (*i) += (*j);
              i->weight(weight);
            }
            target_surface.set_points(displaced_mesh.begin(), displaced_mesh.end());
          }
        }
#endif 

        geometry_ifstr.close();
      }
      nsurf_ptr->add(target_surface);

      std::cout << "Adding NURBS-Volume : " << target_surface.numberofpoints_u() << ", " << target_surface.numberofpoints_v() << std::endl;
    }
  }

  return true;
}


} // namespace gpucast