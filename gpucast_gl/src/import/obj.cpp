/********************************************************************************
*
* Copyright (C) 2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : obj.cpp
*  project    : glpp
*  description:
*
********************************************************************************/
#include "gpucast/gl/import/obj.hpp"

#include <vector>

#include <boost/filesystem/fstream.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/bind.hpp>
#include <boost/optional.hpp>


namespace gpucast { namespace gl {


  ///////////////////////////////////////////////////////////////////////////////
  fileparser_obj::fileparser_obj()
    : fileparser    (),
      _grammar      (),
      _handler      (),
      _stack        ()
  {
    _stack.root = std::shared_ptr<group>(new group);

    _handler["#"]       = std::bind(std::mem_fn(&fileparser_obj::_handle_comment), std::ref(*this), std::placeholders::_1);
    _handler["$"]       = std::bind(std::mem_fn(&fileparser_obj::_handle_comment), std::ref(*this), std::placeholders::_1);

    // vertex data
    _handler["v"]       = std::bind(std::mem_fn(&fileparser_obj::_handle_vertex), std::ref(*this), std::placeholders::_1);
    _handler["vt"]      = std::bind(std::mem_fn(&fileparser_obj::_handle_texcoord), std::ref(*this), std::placeholders::_1);
    _handler["vn"]      = std::bind(std::mem_fn(&fileparser_obj::_handle_normal), std::ref(*this), std::placeholders::_1);
    _handler["vp"]      = std::bind(std::mem_fn(&fileparser_obj::_handle_parameter_space_vertex), std::ref(*this), std::placeholders::_1);
    _handler["cstype"]  = std::bind(std::mem_fn(&fileparser_obj::_handle_spline), std::ref(*this), std::placeholders::_1);
    _handler["deg"]     = std::bind(std::mem_fn(&fileparser_obj::_handle_degree), std::ref(*this), std::placeholders::_1);
    _handler["bmat"]    = std::bind(std::mem_fn(&fileparser_obj::_handle_basis_matrix), std::ref(*this), std::placeholders::_1);
    _handler["step"]    = std::bind(std::mem_fn(&fileparser_obj::_handle_stepsize), std::ref(*this), std::placeholders::_1);

    // elements / faces
    _handler["p"]       = std::bind(std::mem_fn(&fileparser_obj::_handle_point), std::ref(*this), std::placeholders::_1);
    _handler["l"]       = std::bind(std::mem_fn(&fileparser_obj::_handle_line), std::ref(*this), std::placeholders::_1);
    _handler["f"]       = std::bind(std::mem_fn(&fileparser_obj::_handle_face), std::ref(*this), std::placeholders::_1);
    _handler["curv"]    = std::bind(std::mem_fn(&fileparser_obj::_handle_curve), std::ref(*this), std::placeholders::_1);
    _handler["curv2"]   = std::bind(std::mem_fn(&fileparser_obj::_handle_curve2d), std::ref(*this), std::placeholders::_1);
    _handler["surf"]    = std::bind(std::mem_fn(&fileparser_obj::_handle_surface), std::ref(*this), std::placeholders::_1);

     // free form curve/surface
    _handler["parm"]    = std::bind(std::mem_fn(&fileparser_obj::_handle_parameter_value), std::ref(*this), std::placeholders::_1);
    _handler["trim"]    = std::bind(std::mem_fn(&fileparser_obj::_handle_outer_trim_loop), std::ref(*this), std::placeholders::_1);
    _handler["hole"]    = std::bind(std::mem_fn(&fileparser_obj::_handle_inner_trim_loop), std::ref(*this), std::placeholders::_1);
    _handler["scrv"]    = std::bind(std::mem_fn(&fileparser_obj::_handle_special_curve), std::ref(*this), std::placeholders::_1);
    _handler["sp"]      = std::bind(std::mem_fn(&fileparser_obj::_handle_special_point), std::ref(*this), std::placeholders::_1);
    _handler["end"]     = std::bind(std::mem_fn(&fileparser_obj::_handle_end_statement), std::ref(*this), std::placeholders::_1);

    // connectivity
    _handler["con"]     = std::bind(std::mem_fn(&fileparser_obj::_handle_connectivity), std::ref(*this), std::placeholders::_1);

    // grouping
    _handler["g"]       = std::bind(std::mem_fn(&fileparser_obj::_handle_group), std::ref(*this), std::placeholders::_1);
    _handler["s"]       = std::bind(std::mem_fn(&fileparser_obj::_handle_smooth_group), std::ref(*this), std::placeholders::_1);
    _handler["mg"]      = std::bind(std::mem_fn(&fileparser_obj::_handle_merging_group), std::ref(*this), std::placeholders::_1);
    _handler["o"]       = std::bind(std::mem_fn(&fileparser_obj::_handle_object_name), std::ref(*this), std::placeholders::_1);

    // display/render attributes
    _handler["bevel"]   = std::bind(std::mem_fn(&fileparser_obj::_handle_bevel_interpolation), std::ref(*this), std::placeholders::_1);
    _handler["c_interp"]= std::bind(std::mem_fn(&fileparser_obj::_handle_color_interpolation), std::ref(*this), std::placeholders::_1);
    _handler["d_interp"]= std::bind(std::mem_fn(&fileparser_obj::_handle_dissolve_interpolation), std::ref(*this), std::placeholders::_1);
    _handler["lod"]     = std::bind(std::mem_fn(&fileparser_obj::_handle_lod), std::ref(*this), std::placeholders::_1);
    _handler["usemtl"]  = std::bind(std::mem_fn(&fileparser_obj::_handle_use_material), std::ref(*this), std::placeholders::_1);
    _handler["mtllib"]  = std::bind(std::mem_fn(&fileparser_obj::_handle_material_library), std::ref(*this), std::placeholders::_1);
    _handler["shadow_obj"] = std::bind(std::mem_fn(&fileparser_obj::_handle_shadow_casting), std::ref(*this), std::placeholders::_1);
    _handler["trace_obj"] = std::bind(std::mem_fn(&fileparser_obj::_handle_ray_tracing), std::ref(*this), std::placeholders::_1);
    _handler["ctech"]   = std::bind(std::mem_fn(&fileparser_obj::_handle_curve_approximation), std::ref(*this), std::placeholders::_1);
    _handler["stech"]   = std::bind(std::mem_fn(&fileparser_obj::_handle_surface_approximation), std::ref(*this), std::placeholders::_1);

    // general 
    _handler["call"]    = std::bind(std::mem_fn(&fileparser_obj::_handle_call), std::ref(*this), std::placeholders::_1);
  }


  ///////////////////////////////////////////////////////////////////////////////
  fileparser_obj::~fileparser_obj()
  {}


  /////////////////////////////////////////////////////////////////////////////
  /* virtual */ std::shared_ptr<node> 
  fileparser_obj::parse (std::string const& filename)
  {
    // store parent path for material library import
    _stack.parent_path = boost::filesystem::path(filename).parent_path().string();

    // open file and start parsing
    boost::filesystem::ifstream ifstr(filename);

    if (ifstr)
    {
      // copy stream to std::string including white spaces
      ifstr.unsetf(std::ios::skipws);

      while (ifstr.good())
      {
        std::string line;
        std::getline(ifstr, line);
        
        if (!line.empty()) 
        {
          std::string type;
          std::stringstream ss(line);
          ss >> type;

          if (_handler.find(type) != _handler.end()) 
          {
            // apply appropriate handler for line
            _handler[type](line);
          } else {
            std::cerr << "Unknown line syntax in obj. Ignoring line :" << line << std::endl;;
          }
        }
      }
      ifstr.close();

      _create_geode();

      return _stack.root;

    } else {
      throw std::runtime_error("Failed to read " + filename + "\n");
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  void  
  fileparser_obj::_handle_comment( std::string const& /* s */ )
  {}


  // vertex data
  /////////////////////////////////////////////////////////////////////////////
  void  
  fileparser_obj::_handle_vertex( std::string const& s)
  {
    obj::vec4 vertex;
    std::string::const_iterator b = s.begin();
    if (boost::spirit::qi::phrase_parse(b, s.end(), _grammar.vertex_r, boost::spirit::ascii::space, vertex))
    {
      _stack.vertex.push_back(vec4f(vertex.x, vertex.y, vertex.z, vertex.w));
    } else {
     std::cerr << "Wrong syntax for vertex. Ignoring line.\n"; 
    }
  }


  /////////////////////////////////////////////////////////////////////////////
  void  
  fileparser_obj::_handle_texcoord( std::string const& s )
  {
    obj::vec4 texcoord;
    std::string::const_iterator b = s.begin();
    if (boost::spirit::qi::phrase_parse(b, s.end(), _grammar.texcoord_r, boost::spirit::ascii::space, texcoord))
    {
      _stack.texcoord.push_back(vec4f(texcoord.x, texcoord.y, texcoord.z, texcoord.w));
    } else {
      std::cerr << "Wrong syntax for texcoord. Ignoring line.\n"; 
    }
  }


  /////////////////////////////////////////////////////////////////////////////
  void  
  fileparser_obj::_handle_normal( std::string const& s )
  {
    obj::vec4 normal;
    std::string::const_iterator b = s.begin();
    if (boost::spirit::qi::phrase_parse(b, s.end(), _grammar.normal_r, boost::spirit::ascii::space, normal))
    {
      _stack.normal.push_back(vec4f(normal.x, normal.y, normal.z, normal.w) );
    } else {
      std::cerr << "Wrong syntax for normal. Ignoring line.\n"; 
    }
  }


  /////////////////////////////////////////////////////////////////////////////
  void  
  fileparser_obj::_handle_parameter_space_vertex( std::string const& /* s */ )
  {
    std::cerr << "Importing parameter space vertices not supported yet\n";
  }


  /////////////////////////////////////////////////////////////////////////////
  void  
  fileparser_obj::_handle_spline( std::string const& /* s */ )
  {
    std::cerr << "Importing spline not supported yet\n";
  }


  /////////////////////////////////////////////////////////////////////////////
  void 
  fileparser_obj::_handle_degree( std::string const& /* s */ )
  {
    std::cerr << "Importing surface degree not supported yet\n";
  }


  /////////////////////////////////////////////////////////////////////////////
  void 
  fileparser_obj::_handle_basis_matrix( std::string const& /* s */ )
  {
    std::cerr << "Importing basis matrices not supported yet\n";
  }


  /////////////////////////////////////////////////////////////////////////////
  void 
  fileparser_obj::_handle_stepsize( std::string const& /* s */ )
  {
    std::cerr << "Importing step size not supported yet\n";
  }


  // elements/faces
  void  
  fileparser_obj::_handle_point( std::string const& /* s */ )
  {
    std::cerr << "Importing points not supported yet\n";
  }


  /////////////////////////////////////////////////////////////////////////////
  void  
  fileparser_obj::_handle_line( std::string const& /* s */ )
  {
    std::cerr << "Importing lines not supported yet\n";
  }


  /////////////////////////////////////////////////////////////////////////////
  void  
  fileparser_obj::_handle_face( std::string const& s )
  {
    std::vector<std::string> result;
    std::string::const_iterator line_begin = s.begin();
    
    if (boost::spirit::qi::phrase_parse(line_begin, s.end(), _grammar.face_r, boost::spirit::ascii::space, result))
    {
      // make sure there is a geode ready
      if (!_stack.current_geode)
      {
        _stack.current_geode = std::shared_ptr<geode>(new geode);
      }

      std::size_t nvertices_in_face = result.size();
      bool        has_normal        = false;
      bool        has_texcoord      = false;

      // ignore lines! 
      if (nvertices_in_face < 3) 
      {
        // std::cerr << "Omitting line or point element.\n";
        return;
      }

      // process face information
      std::size_t offset = _stack.vertexbuffer.size();
      std::size_t last   = 1;

      for (unsigned i = 2; i < nvertices_in_face; ++i) 
      {
        _stack.indices.push_back(int(offset)       );
        _stack.indices.push_back(int(offset + last));
        _stack.indices.push_back(int(offset + i   ));
        last = i;
      }

      std::vector<vec4f> vertexlist(nvertices_in_face);

      // process all vertex information
      for (unsigned int i = 0; i < nvertices_in_face; ++i)
      {
        std::string const&          vertexinfo  = result[i];
        std::size_t                 nslashes    = std::count(vertexinfo.begin(), vertexinfo.end(), '/');
        std::string::const_iterator b           = vertexinfo.begin();
        obj::vertex_data d;

        switch (nslashes) {
          case 0 :
            boost::spirit::qi::phrase_parse(b, vertexinfo.end(), _grammar.vertex_only_r, boost::spirit::ascii::space, d);
            break;
          case 1 :
            boost::spirit::qi::phrase_parse(b, vertexinfo.end(), _grammar.vertex_texcoord_r, boost::spirit::ascii::space, d);
            break;
          case 2 :
            boost::spirit::qi::phrase_parse(b, vertexinfo.end(), _grammar.vertex_texcoord_normal_r, boost::spirit::ascii::space, d);
            break;
          default :
            std::cerr << "Wrong syntax in face. Too many / per vertex\n";
        };

        // store vertices of polygon for normal computation
        vertexlist[i] = _stack.vertex[d.get<0>() - 1];

        // add given vertex
        _stack.vertexbuffer.push_back     (_stack.vertex    [d.get<0>() - 1]);

        // optionally add texture coordinate
        has_texcoord  = d.get<1>();
        if (has_texcoord) {
          _stack.texcoordbuffer.push_back (_stack.texcoord  [d.get<1>().get() - 1]);
        }

        // optionally add vertex normal
        has_normal    = d.get<2>();
        if (has_normal) {
          _stack.normalbuffer.push_back   (_stack.normal    [d.get<2>().get() - 1]);
        }
      }

      // add normal and texture coordinates if not set
      if (!has_normal)
      {
        vec3f cw_normal = _compute_normal(vertexlist[0].xyz(),
                                          vertexlist[1].xyz(),
                                          vertexlist[2].xyz());

        std::fill_n(std::back_inserter(_stack.normalbuffer), nvertices_in_face, vec4f(cw_normal[0], cw_normal[1], cw_normal[2], 0.0f));
      }
     
    } else {
      std::cerr << "Wrong syntax for face. Ignoring line.\n";  
    }
  }


  /////////////////////////////////////////////////////////////////////////////
  void  
  fileparser_obj::_handle_curve( std::string const& /* s */ )
  {
    std::cerr << "Importing curve not supported yet\n";
  }


  /////////////////////////////////////////////////////////////////////////////
  void  
  fileparser_obj::_handle_curve2d( std::string const& /* s */ )
  {
    std::cerr << "Importing curve 2D not supported yet\n";
  }


  /////////////////////////////////////////////////////////////////////////////
  void  
  fileparser_obj::_handle_surface( std::string const& /* s */ )
  {
    std::cerr << "Importing surface not supported yet\n";
  }


  // free form curve/surface
  /////////////////////////////////////////////////////////////////////////////
  void  
  fileparser_obj::_handle_parameter_value( std::string const& /* s */ )
  {
    std::cerr << "Importing parameter values not supported yet\n";
  }


  /////////////////////////////////////////////////////////////////////////////
  void  
  fileparser_obj::_handle_outer_trim_loop( std::string const& /* s */ )
  {
    std::cerr << "Importing outer trim loops not supported yet\n";
  }


  /////////////////////////////////////////////////////////////////////////////
  void  
  fileparser_obj::_handle_inner_trim_loop( std::string const& /* s */ )
  {
    std::cerr << "Importing inner trim loops not supported yet\n";
  }


  void  
  fileparser_obj::_handle_special_curve( std::string const& /* s */ )
  {
    std::cerr << "Importing special curves not supported yet\n";
  }


  /////////////////////////////////////////////////////////////////////////////
  void  
  fileparser_obj::_handle_special_point( std::string const& /* s */ )
  {
    std::cerr << "Importing special points not supported yet\n";
  }


  /////////////////////////////////////////////////////////////////////////////
  void  
  fileparser_obj::_handle_end_statement( std::string const& /* s */ )
  {
    std::cerr << "Importing end statements not supported yet\n";
  }


  // connectivity
  /////////////////////////////////////////////////////////////////////////////
  void  
  fileparser_obj::_handle_connectivity( std::string const& /* s */ )
  {
    std::cerr << "Importing connectivity not supported yet\n";
  }


  // grouping
  /////////////////////////////////////////////////////////////////////////////
  void  
  fileparser_obj::_handle_group( std::string const& s )
  {
     _create_geode();

    std::vector<std::string> g;
    std::string::const_iterator b = s.begin();
    if (boost::spirit::qi::phrase_parse(b, s.end(), _grammar.group_r, boost::spirit::ascii::space, g))
    {
      _stack.current_group = g;
    } else {
      std::cerr << "Wrong syntax for group. Ignoring line.\n"; 
    }
  }


  /////////////////////////////////////////////////////////////////////////////
  void  
  fileparser_obj::_handle_smooth_group( std::string const& s )
  {
    int smoothgroup;
    std::string::const_iterator b = s.begin();
    if (boost::spirit::qi::phrase_parse(b, s.end(), _grammar.smoothing_group_r, boost::spirit::ascii::space, smoothgroup))
    {
      _stack.smoothing_group = smoothgroup;
    } else {
      std::cerr << "Wrong syntax for smoothing group. Ignoring line.\n"; 
    }
  }


  /////////////////////////////////////////////////////////////////////////////
  void  
  fileparser_obj::_handle_merging_group( std::string const& /* s */ )
  {
    std::cerr << "Importing merging groups not supported yet\n";
  }


  /////////////////////////////////////////////////////////////////////////////
  void  
  fileparser_obj::_handle_object_name( std::string const& /* s */ )
  {
    //std::cerr << "Importing object names not supported yet\n";
  }


  // display/render attributes
  /////////////////////////////////////////////////////////////////////////////
  void  
  fileparser_obj::_handle_bevel_interpolation( std::string const& /* s */ )
  {
    std::cerr << "Importing bevel interpolation not supported yet\n";
  }


  /////////////////////////////////////////////////////////////////////////////
  void  
  fileparser_obj::_handle_color_interpolation( std::string const& /* s */ )
  {
    std::cerr << "Importing color interpolation not supported yet\n";
  }


  /////////////////////////////////////////////////////////////////////////////
  void  
  fileparser_obj::_handle_dissolve_interpolation( std::string const& /* s */ )
  {
    std::cerr << "Importing dissolve interpolation not supported yet\n";
  }


  /////////////////////////////////////////////////////////////////////////////
  void  
  fileparser_obj::_handle_lod( std::string const& /* s */ )
  {
    std::cerr << "Importing level of detail not supported yet\n"; 
  }


  /////////////////////////////////////////////////////////////////////////////
  void  
  fileparser_obj::_handle_use_material( std::string const& s )
  {
    std::string material_name;
    std::string::const_iterator b = s.begin();
    if (boost::spirit::qi::phrase_parse(b, s.end(), _grammar.usemtl_r, boost::spirit::ascii::space, material_name))
    {
      _create_geode();
      _stack.current_material = material_name;
    } else {
      std::cerr << "Wrong syntax for smoothing group. Ignoring line.\n"; 
    }
  }


  /////////////////////////////////////////////////////////////////////////////
  void  
  fileparser_obj::_handle_material_library( std::string const& s )
  {
    std::vector<std::string> material_filenames;
    std::string::const_iterator b = s.begin();
    if (boost::spirit::qi::phrase_parse(b, s.end(), _grammar.mtllib_r, boost::spirit::ascii::space, material_filenames))
    {
      BOOST_FOREACH(std::string const& filename, material_filenames)
      {
        fileparser_mtl mtl_parser;
        mtl_parser.parse(_stack.parent_path + "/" + filename, _stack.materialmap);
      }
    } else {
      std::cerr << "Wrong syntax for smoothing group. Ignoring line.\n"; 
    }
  }


  /////////////////////////////////////////////////////////////////////////////
  void  
  fileparser_obj::_handle_shadow_casting( std::string const& /* s */ )
  {
    std::cerr << "Importing shadow casting not supported yet\n";
  }


  /////////////////////////////////////////////////////////////////////////////
  void   
  fileparser_obj::_handle_ray_tracing( std::string const& /* s */ )
  {
    std::cerr << "Importing ray tracing not supported yet\n";
  }


  /////////////////////////////////////////////////////////////////////////////
  void  
  fileparser_obj::_handle_curve_approximation( std::string const& /* s */ )
  {
    std::cerr << "Importing curve approximation not supported yet\n";
  }


  /////////////////////////////////////////////////////////////////////////////
  void  
  fileparser_obj::_handle_surface_approximation( std::string const& /* s */ )
  {
    std::cerr << "Importing surface approximation not supported yet\n";
  }


  // genaral statement
  /////////////////////////////////////////////////////////////////////////////
  void  
  fileparser_obj::_handle_call( std::string const& /* s */ )
  {
    std::cerr << "Importing calls not supported yet\n";
  }


  ///////////////////////////////////////////////////////////////////////////
  void 
  fileparser_obj::_apply_geometry_to_geode()
  {
    _stack.current_geode->set_mode(GL_TRIANGLES);
    _stack.current_geode->set_material(_stack.materialmap[_stack.current_material]);

    _stack.current_geode->add_attribute_buffer(0, _stack.vertexbuffer,   geode::vertex);

    if (!_stack.normalbuffer.empty())   
      _stack.current_geode->add_attribute_buffer(1, _stack.normalbuffer,   geode::normal);

    if (!_stack.texcoordbuffer.empty()) 
      _stack.current_geode->add_attribute_buffer(2, _stack.texcoordbuffer, geode::texcoord);

    _stack.current_geode->set_indexbuffer (_stack.indices);
  }
  
  
    ///////////////////////////////////////////////////////////////////////////
    void
    fileparser_obj::_create_geode() 
    {
      // create geode only if there is geometry
      if (!_stack.vertexbuffer.empty()) 
      {
        // make sure a geode node is ready
        if (_stack.current_geode) 
        {
          _stack.current_geode->name(std::accumulate(_stack.current_group.begin(), _stack.current_group.end(), std::string("")));
          _apply_geometry_to_geode();
          _stack.root->add(_stack.current_geode);
          _clear_buffers();
        }
        _stack.current_geode = std::shared_ptr<geode>(new geode);
      }
    }


  ///////////////////////////////////////////////////////////////////////////
  void 
  fileparser_obj::_clear_buffers() 
  {
    _stack.vertexbuffer.clear();
    _stack.normalbuffer.clear();
    _stack.texcoordbuffer.clear();
    _stack.indices.clear();
  }

  ///////////////////////////////////////////////////////////////////////////
  vec3f                                   
  fileparser_obj::_compute_normal( vec3f const& v1, vec3f const& v2, vec3f const& v3) const
  {
    return cross(v2-v1, v3-v1);    
  }

} } // namespace gpucast / namespace gl
