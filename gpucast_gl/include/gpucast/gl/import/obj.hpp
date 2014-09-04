/********************************************************************************
*
* Copyright (C) 2007-2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : obj.hpp
*  project    : glpp
*  description:
*
********************************************************************************/
#ifndef GPUCAST_GL_IMPORT_OBJ_GRAMMAR_HPP
#define GPUCAST_GL_IMPORT_OBJ_GRAMMAR_HPP

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
#include <boost/array.hpp>

#include <gpucast/gl/glpp.hpp>
#include <gpucast/gl/import/fileparser.hpp>
#include <gpucast/gl/import/mtl.hpp>


namespace gpucast { namespace gl { namespace obj {

      struct vec4 {
        float x;
        float y;
        float z;
        float w;
      };

      typedef boost::tuple< int,
        boost::optional<int>,
        boost::optional<int>
      > vertex_data;

      struct stack
      {
        typedef fileparser_mtl::material_map_type     material_map_type;

        std::shared_ptr<group>  root;
        std::shared_ptr<geode>  current_geode;

        std::string               parent_path;

        std::vector<gpucast::math::vec4f>        vertex;
        std::vector<gpucast::math::vec4f>        texcoord;
        std::vector<gpucast::math::vec4f>        normal;

        std::vector<gpucast::math::vec4f>        vertexbuffer;
        std::vector<gpucast::math::vec4f>        normalbuffer;
        std::vector<gpucast::math::vec4f>        texcoordbuffer;
        std::vector<int>          indices;

        std::vector<std::string>  current_group;
        std::string               current_material;

        material_map_type         materialmap;

        int                       smoothing_group;
      };

} } } // namespace gpucast / namespace gl / namespace obj

BOOST_FUSION_ADAPT_STRUCT (gpucast::gl::obj::vec4,
                          (float, x)
                          (float, y)
                          (float, z)
                          (float, w)
                          )

namespace gpucast { namespace gl { namespace obj {

      template <typename iterator_t>
      class grammar : public boost::spirit::qi::grammar<iterator_t,
        void(),
        boost::spirit::ascii::space_type>
      {
      public:

        grammar()
          : grammar::base_type(obj_r)
        {
            using boost::spirit::qi::lit;
            using boost::spirit::qi::lexeme;
            using boost::spirit::qi::string;
            using boost::spirit::qi::float_;
            using boost::spirit::qi::int_;
            using boost::spirit::qi::eps;
            using boost::spirit::standard::char_;
            using boost::spirit::ascii::space;

            using boost::phoenix::at_c;
            using boost::phoenix::push_back;

            using namespace boost::spirit::qi::labels;

            vertex_r = eps[at_c<3>(_val) = 1.0f] >>
              lit('v') >> float_[at_c<0>(_val) = boost::spirit::qi::_1]
              >> float_[at_c<1>(_val) = boost::spirit::qi::_1]
              >> float_[at_c<2>(_val) = boost::spirit::qi::_1]
              >> -(float_[at_c<3>(_val) = boost::spirit::qi::_1]);

            normal_r = eps[at_c<3>(_val) = 0.0f] >>
              lit("vn") >> float_[at_c<0>(_val) = boost::spirit::qi::_1]
              >> float_[at_c<1>(_val) = boost::spirit::qi::_1]
              >> float_[at_c<2>(_val) = boost::spirit::qi::_1];

            texcoord_r = eps[at_c<3>(_val) = 0.0f] >>
              lit("vt") >> float_[at_c<0>(_val) = boost::spirit::qi::_1]
              >> -(float_[at_c<1>(_val) = boost::spirit::qi::_1])
              >> -(float_[at_c<2>(_val) = boost::spirit::qi::_1]);

            // reading face and vertex information
            face_r %= lit('f') >> +(identifier_r);

            vertex_only_r = int_[at_c<0>(_val) = boost::spirit::qi::_1];

            vertex_texcoord_r = int_[at_c<0>(_val) = boost::spirit::qi::_1]
              >> lit('/')
              >> int_[at_c<1>(_val) = boost::spirit::qi::_1];

            vertex_texcoord_normal_r = int_[at_c<0>(_val) = boost::spirit::qi::_1]
              >> lit('/')
              >> -int_[at_c<1>(_val) = boost::spirit::qi::_1]
              >> lit('/')
              >> -int_[at_c<2>(_val) = boost::spirit::qi::_1];

            // group and text
            group_r %= lit('g') >> +(identifier_r);

            identifier_r %= lexeme[+(char_ - space)];

            smoothing_group_r = lit('s') >> (int_[_val = boost::spirit::qi::_1]
              | lit("off")[_val = 0]
              );

            // materials
            mtllib_r = lit("mtllib") >> +(identifier_r[push_back(_val, boost::spirit::qi::_1)]);
            usemtl_r %= lit("usemtl") >> identifier_r;

          }

      public: // member

        boost::spirit::qi::rule<iterator_t, void(), boost::spirit::ascii::space_type>   obj_r;
        boost::spirit::qi::rule<iterator_t, vec4(), boost::spirit::ascii::space_type>   vertex_r;
        boost::spirit::qi::rule<iterator_t, vec4(), boost::spirit::ascii::space_type>   texcoord_r;
        boost::spirit::qi::rule<iterator_t, vec4(), boost::spirit::ascii::space_type>   normal_r;
        boost::spirit::qi::rule<iterator_t, void(), boost::spirit::ascii::space_type>   comment_r;

        boost::spirit::qi::rule<iterator_t, std::vector<std::string>(), boost::spirit::ascii::space_type>   face_r;
        boost::spirit::qi::rule<iterator_t, vertex_data(), boost::spirit::ascii::space_type>   vertex_only_r;
        boost::spirit::qi::rule<iterator_t, vertex_data(), boost::spirit::ascii::space_type>   vertex_texcoord_r;
        boost::spirit::qi::rule<iterator_t, vertex_data(), boost::spirit::ascii::space_type>   vertex_texcoord_normal_r;

        boost::spirit::qi::rule<iterator_t, std::vector<std::string>(), boost::spirit::ascii::space_type>   group_r;
        boost::spirit::qi::rule<iterator_t, std::string(), boost::spirit::ascii::space_type>   identifier_r;
        boost::spirit::qi::rule<iterator_t, int(), boost::spirit::ascii::space_type>   smoothing_group_r;
        boost::spirit::qi::rule<iterator_t, std::vector<std::string>(), boost::spirit::ascii::space_type>   mtllib_r;
        boost::spirit::qi::rule<iterator_t, std::string(), boost::spirit::ascii::space_type>   usemtl_r;
      };

} } } // namespace gpucast / namespace gl / namespace obj


namespace gpucast { namespace gl {

  class GPUCAST_GL fileparser_obj : public fileparser, public boost::noncopyable
  {
  public : // typedef/enum

    typedef std::function<void(std::string const&)>     handler_fun_type;

  public :

    fileparser_obj                                        ();
    virtual ~fileparser_obj                               ();
    virtual std::shared_ptr<node>         parse         ( std::string const& filename );

  private :

    void  _apply_geometry_to_geode        ();
    void  _create_geode                   ();
    void  _clear_buffers                  ();
    gpucast::math::vec3f _compute_normal                 ( gpucast::math::vec3f const& v1, gpucast::math::vec3f const& v2, gpucast::math::vec3f const& v3) const;

  private : // handler

    // comments
    void  _handle_comment                 ( std::string const& );

    // vertex data
    void  _handle_vertex                  ( std::string const& );
    void  _handle_texcoord                ( std::string const& );
    void  _handle_normal                  ( std::string const& );
    void  _handle_parameter_space_vertex  ( std::string const& );
    void  _handle_spline                  ( std::string const& );
    void  _handle_degree                  ( std::string const& );
    void  _handle_basis_matrix            ( std::string const& );
    void  _handle_stepsize                ( std::string const& );

    // elements/faces
    void  _handle_point                   ( std::string const& );
    void  _handle_line                    ( std::string const& );
    void  _handle_face                    ( std::string const& );
    void  _handle_curve                   ( std::string const& );
    void  _handle_curve2d                 ( std::string const& );
    void  _handle_surface                 ( std::string const& );

    // free form curve/surface
    void  _handle_parameter_value         ( std::string const& );
    void  _handle_outer_trim_loop         ( std::string const& );
    void  _handle_inner_trim_loop         ( std::string const& );
    void  _handle_special_curve           ( std::string const& );
    void  _handle_special_point           ( std::string const& );
    void  _handle_end_statement           ( std::string const& );

    // connectivity
    void  _handle_connectivity            ( std::string const& );

    // grouping
    void  _handle_group                   ( std::string const& );
    void  _handle_smooth_group            ( std::string const& );
    void  _handle_merging_group           ( std::string const& );
    void  _handle_object_name             ( std::string const& );

    // display/render attributes
    void  _handle_bevel_interpolation     ( std::string const& );
    void  _handle_color_interpolation     ( std::string const& );
    void  _handle_dissolve_interpolation  ( std::string const& );
    void  _handle_lod                     ( std::string const& );
    void  _handle_use_material            ( std::string const& );
    void  _handle_material_library        ( std::string const& );
    void  _handle_shadow_casting          ( std::string const& );
    void  _handle_ray_tracing             ( std::string const& );
    void  _handle_curve_approximation     ( std::string const& );
    void  _handle_surface_approximation   ( std::string const& );

    // genaral statement
    void  _handle_call                    ( std::string const& );


  private : // attributes

    obj::grammar<std::string::const_iterator>            _grammar;
    std::unordered_map<std::string, handler_fun_type >   _handler;
    obj::stack                                           _stack;

  };

} } // namespace gpucast / namespace gl


#endif // GPUCAST_GL_IMPORT_OBJ_GRAMMAR_HPP
