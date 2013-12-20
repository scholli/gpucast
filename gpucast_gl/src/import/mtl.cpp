/********************************************************************************
*
* Copyright (C) 2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : mtl.cpp
*  project    : glpp
*  description:
*
********************************************************************************/
#include "gpucast/gl/import/mtl.hpp"

#include <vector>
#include <functional>

namespace gpucast { namespace gl {

  ///////////////////////////////////////////////////////////////////////////////
  fileparser_mtl::fileparser_mtl()
    : _grammar          (),
      _handler          (),
      _current_material ()
  {
    // comments
    _handler["#"] = std::bind(&fileparser_mtl::_handle_comment, std::ref(*this), std::placeholders::_1, std::placeholders::_2);

    _handler["newmtl"]  = std::bind(&fileparser_mtl::_handle_name, std::ref(*this), std::placeholders::_1, std::placeholders::_2);
    _handler["Ka"]      = std::bind(&fileparser_mtl::_handle_ambient, std::ref(*this),std::placeholders::_1,std::placeholders::_2);
    _handler["Kd"]      = std::bind(&fileparser_mtl::_handle_diffuse, std::ref(*this),std::placeholders::_1,std::placeholders::_2);
    _handler["Ks"]      = std::bind(&fileparser_mtl::_handle_specular, std::ref(*this),std::placeholders::_1,std::placeholders::_2);
    _handler["Tr"]      = std::bind(&fileparser_mtl::_handle_transparency, std::ref(*this),std::placeholders::_1,std::placeholders::_2);
    _handler["d"]       = std::bind(&fileparser_mtl::_handle_transparency, std::ref(*this),std::placeholders::_1,std::placeholders::_2);
    _handler["illum"]   = std::bind(&fileparser_mtl::_handle_specularity, std::ref(*this),std::placeholders::_1,std::placeholders::_2);
    _handler["Ns"]      = std::bind(&fileparser_mtl::_handle_shininess, std::ref(*this),std::placeholders::_1,std::placeholders::_2);

    // maps
    _handler["map_Ka"]  = std::bind(&fileparser_mtl::_handle_ambientmap, std::ref(*this),std::placeholders::_1,std::placeholders::_2);
    _handler["map_kA"]  = std::bind(&fileparser_mtl::_handle_ambientmap, std::ref(*this),std::placeholders::_1,std::placeholders::_2);
    _handler["map_ka"]  = std::bind(&fileparser_mtl::_handle_ambientmap, std::ref(*this),std::placeholders::_1,std::placeholders::_2);

    _handler["map_Kd"]  = std::bind(&fileparser_mtl::_handle_diffusemap, std::ref(*this),std::placeholders::_1,std::placeholders::_2);
    _handler["map_kD"]  = std::bind(&fileparser_mtl::_handle_diffusemap, std::ref(*this),std::placeholders::_1,std::placeholders::_2);
    _handler["map_kd"]  = std::bind(&fileparser_mtl::_handle_diffusemap, std::ref(*this),std::placeholders::_1,std::placeholders::_2);

    _handler["map_Ks"]  = std::bind(&fileparser_mtl::_handle_specularmap, std::ref(*this),std::placeholders::_1,std::placeholders::_2);
    _handler["map_ks"]  = std::bind(&fileparser_mtl::_handle_specularmap, std::ref(*this),std::placeholders::_1,std::placeholders::_2);
    _handler["map_kS"]  = std::bind(&fileparser_mtl::_handle_specularmap, std::ref(*this),std::placeholders::_1,std::placeholders::_2);

    _handler["map_Bump"]= std::bind(&fileparser_mtl::_handle_bumpmap, std::ref(*this),std::placeholders::_1,std::placeholders::_2);
    _handler["map_bump"]= std::bind(&fileparser_mtl::_handle_bumpmap, std::ref(*this),std::placeholders::_1,std::placeholders::_2);
    _handler["bump"]    = std::bind(&fileparser_mtl::_handle_bumpmap, std::ref(*this),std::placeholders::_1,std::placeholders::_2);

    _handler["map_d"]       = std::bind(&fileparser_mtl::_handle_bumpmap, std::ref(*this),std::placeholders::_1,std::placeholders::_2);
    _handler["map_opacity"] = std::bind(&fileparser_mtl::_handle_bumpmap, std::ref(*this),std::placeholders::_1,std::placeholders::_2);
  }


  ///////////////////////////////////////////////////////////////////////////////
  fileparser_mtl::~fileparser_mtl()
  {}


  /////////////////////////////////////////////////////////////////////////////
  void 
  fileparser_mtl::parse (std::string const& filename, fileparser_mtl::material_map_type& m )
  {
    std::fstream ifstr(filename, std::ios::in);

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

          if (!type.empty()) // skip empty lines
          {
            if (type[0] != '#') // skip comments
            {
              if (_handler.find(type) != _handler.end()) 
              {
                // apply appropriate handler for line
                _handler[type](line, m);
              } else {
                std::cerr << "Ignoring unknown line syntax : " <<  line << std::endl;
              }
            }
          }
        }
      }
      ifstr.close();

    } else {
      std::cerr << "Failed to read " << filename << std::endl;
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  void  
  fileparser_mtl::_handle_comment( std::string const& s, fileparser_mtl::material_map_type& m  )
  {}


  // vertex data
  /////////////////////////////////////////////////////////////////////////////
  void  
  fileparser_mtl::_handle_name( std::string const& s, fileparser_mtl::material_map_type& m )
  {
    std::string current_material;
    std::string::const_iterator b = s.begin();
    if (boost::spirit::qi::phrase_parse(b, s.end(), _grammar.name_r, boost::spirit::ascii::space, current_material))
    {
      _current_material = current_material;
      material new_material;
      m.insert(std::make_pair(_current_material, new_material));
    } 
  }


  /////////////////////////////////////////////////////////////////////////////
  void  
  fileparser_mtl::_handle_ambient( std::string const& s, fileparser_mtl::material_map_type& m  )
  {
    mtl::color c;
    std::string::const_iterator b = s.begin();
    if (boost::spirit::qi::phrase_parse(b, s.end(), _grammar.ambient_r, boost::spirit::ascii::space, c))
    {
      m[_current_material].ambient = vec4f(c.r, c.g, c.b, 1.0);
    } 
  }


  /////////////////////////////////////////////////////////////////////////////
  void  
  fileparser_mtl::_handle_diffuse( std::string const& s, fileparser_mtl::material_map_type& m  )
  {
    mtl::color c;
    std::string::const_iterator b = s.begin();
    if (boost::spirit::qi::phrase_parse(b, s.end(), _grammar.diffuse_r, boost::spirit::ascii::space, c))
    {
      m[_current_material].diffuse = vec4f(c.r, c.g, c.b, 1.0);
    } 
  }


  /////////////////////////////////////////////////////////////////////////////
  void  
  fileparser_mtl::_handle_specular( std::string const& s, fileparser_mtl::material_map_type& m  )
  {
    mtl::color c;
    std::string::const_iterator b = s.begin();
    if (boost::spirit::qi::phrase_parse(b, s.end(), _grammar.specular_r, boost::spirit::ascii::space, c))
    {
      m[_current_material].specular = vec4f(c.r, c.g, c.b, 1.0);
    }
  }


  /////////////////////////////////////////////////////////////////////////////
  void  
  fileparser_mtl::_handle_specularity( std::string const& s, fileparser_mtl::material_map_type& m  )
  {
    int illum;
    std::string::const_iterator b = s.begin();
    if (boost::spirit::qi::phrase_parse(b, s.end(), _grammar.illum_r, boost::spirit::ascii::space, illum))
    {
      switch (illum) {
        // turn off specularity
        case 1 : 
          m[_current_material].specular = vec4f(0.0, 0.0, 0.0, 0.0);
          break;
        case 2 :
          // turn on specularity - do nothing
          break;
        default :
          // do nothing
          break;
      };
    }
  }


  /////////////////////////////////////////////////////////////////////////////
  void  
  fileparser_mtl::_handle_transparency( std::string const& s, fileparser_mtl::material_map_type& m  )
  {
    float transparency;
    std::string::const_iterator b = s.begin();
    if (boost::spirit::qi::phrase_parse(b, s.end(), _grammar.transparency_r, boost::spirit::ascii::space, transparency))
    {
      m[_current_material].opacity = transparency;
    }
  }


  /////////////////////////////////////////////////////////////////////////////
  void  
  fileparser_mtl::_handle_shininess( std::string const& s, fileparser_mtl::material_map_type& m  )
  {
    float shininess;
    std::string::const_iterator b = s.begin();
    if (boost::spirit::qi::phrase_parse(b, s.end(), _grammar.shininess_r, boost::spirit::ascii::space, shininess))
    {
      m[_current_material].shininess = shininess;
    }
  }


  /////////////////////////////////////////////////////////////////////////////
  void 
  fileparser_mtl::_handle_ambientmap( std::string const& s, fileparser_mtl::material_map_type& m  )
  {
    std::cerr << "Importing ambient map not supported yet\n";
  }


  /////////////////////////////////////////////////////////////////////////////
  void 
  fileparser_mtl::_handle_diffusemap( std::string const& s, fileparser_mtl::material_map_type& m  )
  {
    std::cerr << "Importing diffuse map not supported yet\n";
  }


  /////////////////////////////////////////////////////////////////////////////
  void 
  fileparser_mtl::_handle_specularmap( std::string const& s, fileparser_mtl::material_map_type& m  )
  {
    std::cerr << "Importing specular map not supported yet\n";
  }


  /////////////////////////////////////////////////////////////////////////////
  void  
  fileparser_mtl::_handle_bumpmap( std::string const& s, fileparser_mtl::material_map_type& m  )
  {
    std::cerr << "Importing bump map supported yet\n";
  }

  /////////////////////////////////////////////////////////////////////////////
  void  
  fileparser_mtl::_handle_opacitymap( std::string const& s, fileparser_mtl::material_map_type& m  )
  {
    std::cerr << "Importing opacity map supported yet\n";
  }


} } // namespace gpucast / namespace gl
