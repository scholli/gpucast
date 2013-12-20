/********************************************************************************
*
* Copyright (C) 2009 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : transferfunction.hpp
*  project    : glpp
*  description:
*
********************************************************************************/
#ifndef GPUCAST_GL_TRANSFERFUNCTION_HPP
#define GPUCAST_GL_TRANSFERFUNCTION_HPP

// header, system
#include <map>

// header, project
#include <gpucast/gl/glpp.hpp>

namespace gpucast { namespace gl {

class piecewise_linear 
{
public :

  template <typename map_type>
  typename map_type::mapped_type operator()( map_type const& nodes, float value )
  {
    assert(nodes.size() >= 2);

    value = std::min ( std::max ( value, 0.0f ), 1.0f );
    unsigned char as_uchar = unsigned char( value * std::numeric_limits<unsigned char>::max() );

    typename map_type::const_iterator first  = nodes.begin();
    typename map_type::const_iterator last   = nodes.begin();
    std::advance(last, 1);

    while ( last != nodes.end() )
    {
      // if as_uchar is in range -> interpolate between 
      if (  as_uchar >= first->first && 
            as_uchar <= last->first ) 
      {
        float a = float(as_uchar - first->first) / float(last->first - first->first);

        return (1.0f - a) * first->second + a * last->second ;
      }
      ++first; ++last;
    }
      
    throw std::runtime_error("transferfunction::piecewise_linear::operator(): position out of range.\n");
  }
};

template <typename value_t>
class transferfunction
{
public : // typedefs

  typedef value_t                               value_type;
  typedef std::map<unsigned char, value_type>   map_type;

public : // interpolation types

public : // c'tor / d'tor

  transferfunction                     ();
  virtual ~transferfunction            ();

public : // methods

  bool              get                ( unsigned char      pos,
                                         value_type&        value ) const;

  void              set                ( unsigned char      pos, 
                                         value_type const&  value );

  bool              remove             ( unsigned char      pos );

  template <typename interpolation_t>
  value_type        evaluate           ( float t, interpolation_t ) const;

  template <typename container_t, typename interpolation_t>
  void              evaluate           ( int nsamples, container_t&, interpolation_t ) const;

private : // attributes

  map_type          _nodemap;

};

} } // namespace gpucast / namespace gl

#include <gpucast/gl/util/transferfunction_impl.hpp>

#endif // GPUCAST_GL_TRANSFERFUNCTION_HPP
