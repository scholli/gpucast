/********************************************************************************
*
* Copyright (C) 2007-2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : mtl.hpp
*  project    : glpp
*  description:
*
********************************************************************************/
#ifndef GPUCAST_GL_IMPORT_MTL_GRAMMAR_HPP
#define GPUCAST_GL_IMPORT_MTL_GRAMMAR_HPP

#if WIN32
  #pragma warning(disable:4512) // boost::spirit::qi ambiguity
  #pragma warning(disable:4100) // boost::spirit::qi unreferenced parameter
  #pragma warning(disable:4127) // boost::spirit::qi conditional expression is constant
#endif

#include <boost/spirit/include/qi.hpp>
#include <boost/fusion/adapted.hpp>
#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>
#include <boost/spirit/include/phoenix_fusion.hpp>
#include <boost/spirit/include/phoenix_stl.hpp>
#include <boost/variant/recursive_variant.hpp>
#include <boost/foreach.hpp>
#include <boost/noncopyable.hpp>
#include <boost/array.hpp>

#include <boost/filesystem/fstream.hpp>
#include <unordered_map>

#include <gpucast/gl/glpp.hpp>
#include <gpucast/gl/import/fileparser.hpp>
#include <gpucast/gl/graph/geode.hpp>
#include <gpucast/gl/util/material.hpp>


namespace gpucast { namespace gl {
    namespace mtl {

      struct color {
        float r;
        float g;
        float b;
      };
    }
} } // namespace gpucast / namespace gl

BOOST_FUSION_ADAPT_STRUCT (gpucast::gl::mtl::color,
                          (float, r)
                          (float, g)
                          (float, b)
                          )

namespace gpucast { namespace gl {
  namespace mtl {

    template <typename iterator_t>
    class grammar : public boost::spirit::qi::grammar<iterator_t,
      void(),
      boost::spirit::ascii::space_type>
    {
    public:

      grammar()
        : grammar::base_type(comment_r)
      {
          using boost::spirit::qi::lit;
          using boost::spirit::qi::float_;
          using boost::spirit::qi::int_;
          using boost::spirit::qi::lexeme;
          using boost::spirit::ascii::space;
          using boost::spirit::standard::char_;

          using boost::phoenix::at_c;

          using namespace boost::spirit::qi::labels;

          comment_r = lexeme[lit('#') >> *char_];

          name_r %= lit("newmtl") >> +char_;

          identifier_r %= lexeme[+(char_ - space)];

          ambient_r = (lit("Ka") | lit("ka") | lit("kA"))
            >> float_[at_c<0>(_val) = boost::spirit::qi::_1]
            >> float_[at_c<1>(_val) = boost::spirit::qi::_1]
            >> float_[at_c<2>(_val) = boost::spirit::qi::_1]
            >> -comment_r
            ;

          diffuse_r = (lit("Kd") | lit("kd") | lit("kD"))
            >> float_[at_c<0>(_val) = boost::spirit::qi::_1]
            >> float_[at_c<1>(_val) = boost::spirit::qi::_1]
            >> float_[at_c<2>(_val) = boost::spirit::qi::_1]
            >> -comment_r
            ;

          specular_r = (lit("Ks") | lit("ks") | lit("kS"))
            >> float_[at_c<0>(_val) = boost::spirit::qi::_1]
            >> float_[at_c<1>(_val) = boost::spirit::qi::_1]
            >> float_[at_c<2>(_val) = boost::spirit::qi::_1]
            >> -comment_r
            ;

          shininess_r %= lit("Ns") >> float_ >> -comment_r;

          transparency_r %= (lit('d') | lit("Tr"))
            >> float_ >> -comment_r;

          illum_r %= lit("illum") >> int_ >> -comment_r;

          bumpmap_r %= lit("map_bump")
            >> identifier_r;

          opacitymap_r %= lit("map_d") | lit("map_opacity")
            >> identifier_r;
        }

    public: // member

      boost::spirit::qi::rule<iterator_t, void(), boost::spirit::ascii::space_type>   comment_r;
      boost::spirit::qi::rule<iterator_t, std::string(), boost::spirit::ascii::space_type>   name_r;
      boost::spirit::qi::rule<iterator_t, std::string(), boost::spirit::ascii::space_type>   identifier_r;

      boost::spirit::qi::rule<iterator_t, color(), boost::spirit::ascii::space_type>   ambient_r;
      boost::spirit::qi::rule<iterator_t, color(), boost::spirit::ascii::space_type>   diffuse_r;
      boost::spirit::qi::rule<iterator_t, color(), boost::spirit::ascii::space_type>   specular_r;

      boost::spirit::qi::rule<iterator_t, float(), boost::spirit::ascii::space_type>   shininess_r;
      boost::spirit::qi::rule<iterator_t, float(), boost::spirit::ascii::space_type>   transparency_r;
      boost::spirit::qi::rule<iterator_t, int(), boost::spirit::ascii::space_type>   illum_r;

      boost::spirit::qi::rule<iterator_t, std::string(), boost::spirit::ascii::space_type>   opacitymap_r;
      boost::spirit::qi::rule<iterator_t, std::string(), boost::spirit::ascii::space_type>   bumpmap_r;
    };

  } // namespace mtl
} } // namespace gpucast / namespace gl  

namespace gpucast { namespace gl {

  class GPUCAST_GL fileparser_mtl : public boost::noncopyable
  {
  public : // typedef/enum

    typedef std::unordered_map<std::string, material>                     material_map_type;
    typedef std::function<void(std::string const&, material_map_type&)>   handler_fun_type;

  public :

    fileparser_mtl                                        ();
    virtual ~fileparser_mtl                               ();

    void        parse                                     ( std::string const& filename,
                                                            material_map_type& material_map );

  private : // handler

    // comments
    void  _handle_comment                 ( std::string const&, material_map_type& );
    void  _handle_name                    ( std::string const&, material_map_type& );
    void  _handle_ambient                 ( std::string const&, material_map_type& );
    void  _handle_diffuse                 ( std::string const&, material_map_type& );
    void  _handle_specular                ( std::string const&, material_map_type& );
    void  _handle_specularity             ( std::string const&, material_map_type& );
    void  _handle_transparency            ( std::string const&, material_map_type& );
    void  _handle_shininess               ( std::string const&, material_map_type& );

    void  _handle_ambientmap              ( std::string const&, material_map_type& );
    void  _handle_diffusemap              ( std::string const&, material_map_type& );
    void  _handle_specularmap             ( std::string const&, material_map_type& );
    void  _handle_bumpmap                 ( std::string const&, material_map_type& );
    void  _handle_opacitymap              ( std::string const&, material_map_type& );


  private : // attributes

    gpucast::gl::mtl::grammar<std::string::const_iterator>              _grammar;
    std::unordered_map<std::string, handler_fun_type >   _handler;
    std::string                                            _current_material;

  };

} } // namespace gpucast / namespace gl


#endif // GPUCAST_GL_IMPORT_MTL_GRAMMAR_HPP
