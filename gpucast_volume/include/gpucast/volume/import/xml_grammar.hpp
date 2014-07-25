/********************************************************************************
*
* Copyright (C) 2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : xml_grammar.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_XML_GRAMMAR_HPP
#define GPUCAST_XML_GRAMMAR_HPP

#if WIN32
  #pragma warning(disable:4512) // boost warnings
#endif

#include <string>
#include <vector>
#include <list>

#include <boost/spirit/include/qi.hpp>
#include <boost/fusion/adapted.hpp>
#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>
#include <boost/spirit/include/phoenix_fusion.hpp>
#include <boost/spirit/include/phoenix_stl.hpp>
#include <boost/variant/recursive_variant.hpp>

#include <gpucast/volume/nurbsvolumeobject.hpp>
#include <gpucast/math/parametric/point.hpp>

namespace gpucast {
  namespace xml  {

    struct controlpoint_attribute
    {
      std::string         name;
      int                 elements;
      std::vector<double> data;
    };

    struct controlpoint
    {
      std::list<controlpoint_attribute> data;
    };

    struct nurbsvolumeobject
    {
      int                       degree_u;
      int                       degree_v;
      int                       degree_w;

      std::vector<double>       knots_u;
      std::vector<double>       knots_v;
      std::vector<double>       knots_w;

      std::vector<controlpoint> points;
    };

  } // namespace xml
} // namespace gpucast

BOOST_FUSION_ADAPT_STRUCT(gpucast::xml::nurbsvolumeobject,
                          (int, degree_u)
                          (int, degree_v)
                          (int, degree_w)
                          (std::vector<double>,       knots_u)
                          (std::vector<double>,       knots_v)
                          (std::vector<double>,       knots_w)
                          (std::vector<gpucast::xml::controlpoint>, points)
                         )

                         
BOOST_FUSION_ADAPT_STRUCT(gpucast::xml::controlpoint_attribute,
                           (std::string,       name)
                           (int,               elements)
                           (std::vector<double>, data)
                         )

BOOST_FUSION_ADAPT_STRUCT(gpucast::xml::controlpoint,
                          (std::list<gpucast::xml::controlpoint_attribute>, data)
                         )


namespace gpucast {
  namespace xml   {

    template <typename iterator_t>
    class grammar : public boost::spirit::qi::grammar<iterator_t,
                                                      std::vector<nurbsvolumeobject>(),
                                                      //controlpoint_attribute(),
                                                      //std::string(),
                                                      //int(),
                                                      boost::spirit::ascii::space_type>
    {
    public :

      grammar()
        : grammar::base_type(xml_r)
          //grammar::base_type(controlpoint_attribute_r)
          //grammar::base_type(size_r)
          //grammar::base_type(name_r)
      {
        using boost::spirit::qi::lit;
        using boost::spirit::qi::_1;
        using boost::spirit::qi::eps;
        using boost::spirit::qi::lexeme;
        using boost::spirit::standard::char_;
        using boost::spirit::qi::int_;
        using boost::spirit::qi::double_;
        using boost::spirit::ascii::space;
        using boost::spirit::ascii::alnum;
        using boost::spirit::ascii::alpha;
        using boost::spirit::ascii::digit;

        using boost::phoenix::at_c;
        using boost::phoenix::push_back;

        using namespace boost::spirit::qi::labels;

        xml_r           %= *nurbsvolumeobject_r;

        number_r        %= '"' >> int_ >> '"';
        type_r          %= lit("type") >> '=' >> '"' >> +(char_ - '"') >> '"';
        dim_r           %= lit("dim")  >> '=' >> number_r;
        size_r          %= lit("size") >> '=' >> number_r;
        id_r            %= lit("id")   >> '=' >> number_r;
        name_r          %= lexeme[+(char_ - '>' - space - '>')];

        nurbsvolumeobject_r   = (lit("<NurbsObject") || lit("<NurbsPatch"))  >> id_r >> lit(">")
                          >> dimension_r
                          >> degree_r         [ at_c<0>(_val) = _1 ]
                          >> degree_r         [ at_c<1>(_val) = _1 ]
                          >> degree_r         [ at_c<2>(_val) = _1 ]
                          >> knotvector_r     [ at_c<3>(_val) = _1 ]
                          >> knotvector_r     [ at_c<4>(_val) = _1 ]
                          >> knotvector_r     [ at_c<5>(_val) = _1 ]
                          >> controlarray_r   [ at_c<6>(_val) = _1 ]
                          >> (lit("</NurbsObject>") || lit("</NurbsPatch>"));

        dimension_r     = lit("<Dimension")
                          >> size_r >> type_r >> lit(">")
                          >> int_
                          >> lit("</Dimension>");

        degree_r        = lit("<Degree")
                          >> dim_r
                          >> size_r
                          >> type_r
                          >> lit(">")
                          >> int_ [_val = _1]
                          >> lit("</Degree>");

        knotvector_r    = lit("<KnotVector")
                          >> dim_r
                          >> size_r
                          >> type_r
                          >> lit(">")
                          >> +double_ [push_back(_val, _1)]
                          >> lit("</KnotVector>");

        controlarray_r  = lit("<ControlArray>") >>
                            + ( lit("<Sequence>")
                                >> +( lit("<Sequence>")
                                      >> + ( lit("<Sequence>")
                                            >> +controlpoint_r [push_back(_val, _1)]
                                            >> lit("</Sequence>")
                                           )
                                      >> lit("</Sequence>")
                                    )
                                >> lit("</Sequence>")
                              )
                          >> lit("</ControlArray>")
                          ;

        controlpoint_r  = lit("<ControlPoint") >> id_r >> lit(">")
                          >> +controlpoint_attribute_r [push_back(at_c<0>(_val), _1)]
                          >> lit("</ControlPoint>")
                          ;


        controlpoint_attribute_r =     lit("<")
                                    >> name_r       [ at_c<0>(_val) = _1 ] 
                                    >> -space
                                    >> size_r       [ at_c<1>(_val) = _1 ] 
                                    >> type_r 
                                    >> lit(">")
                                    >> +double_     [ push_back(at_c<2>(_val), _1) ]
                                    >> lit("</")
                                    >> name_r
                                    >> lit(">")
                                    ;
      }

      ~grammar() {};



    public : // methods

      double a_;

    private : // member

      boost::spirit::qi::rule<iterator_t,       std::vector<nurbsvolumeobject>(),                boost::spirit::ascii::space_type>   xml_r;
      boost::spirit::qi::rule<iterator_t,       nurbsvolumeobject(),                             boost::spirit::ascii::space_type>   nurbsvolumeobject_r;
      boost::spirit::qi::rule<iterator_t,       int(),                                           boost::spirit::ascii::space_type>   number_r;
      boost::spirit::qi::rule<iterator_t,       std::string(),                                   boost::spirit::ascii::space_type>   type_r;
      boost::spirit::qi::rule<iterator_t,       std::string(),                                   boost::spirit::ascii::space_type>   name_r;

      boost::spirit::qi::rule<iterator_t,       void(),                                          boost::spirit::ascii::space_type>   dimension_r;
      boost::spirit::qi::rule<iterator_t,       int(),                                           boost::spirit::ascii::space_type>   degree_r;
      boost::spirit::qi::rule<iterator_t,       int(),                                           boost::spirit::ascii::space_type>   size_r;
      boost::spirit::qi::rule<iterator_t,       int(),                                           boost::spirit::ascii::space_type>   dim_r;
      boost::spirit::qi::rule<iterator_t,       int(),                                           boost::spirit::ascii::space_type>   id_r;
      boost::spirit::qi::rule<iterator_t,       std::vector<double>(),                           boost::spirit::ascii::space_type>   knotvector_r;

      boost::spirit::qi::rule<iterator_t,       controlpoint_attribute(),                        boost::spirit::ascii::space_type>   controlpoint_attribute_r;
      boost::spirit::qi::rule<iterator_t,       controlpoint(),                                  boost::spirit::ascii::space_type>   controlpoint_r;
      boost::spirit::qi::rule<iterator_t,       std::vector<controlpoint>(),                     boost::spirit::ascii::space_type>   controlarray_r;
    };

  } // namespace xml
} // namespace gpucast


#endif // GPUCAST_XML_GRAMMAR_HPP
