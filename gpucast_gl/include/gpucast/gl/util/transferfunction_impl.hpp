/********************************************************************************
*
* Copyright (C) 2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : transferfunction_impl.hpp
*  project    : glpp
*  description:
*
********************************************************************************/

// header, system
#include <limits>

#include <boost/numeric/conversion/bounds.hpp>

// header, project

namespace gpucast { namespace gl {

  ////////////////////////////////////////////////////////////////////////////////
  template <typename value_t>
  inline transferfunction<value_t>::transferfunction()
    : _nodemap()
  {
    set( std::numeric_limits<unsigned char>::max(), value_t() );
    set( std::numeric_limits<unsigned char>::min(), value_t() );
  }


  ////////////////////////////////////////////////////////////////////////////////
  template <typename value_t>
  inline transferfunction<value_t>::~transferfunction()
  {}


  ////////////////////////////////////////////////////////////////////////////////
  template <typename value_t>
  inline bool
  transferfunction<value_t>::get ( unsigned char pos, value_t& v ) const
  {
    map_type::const_iterator i = _nodemap.find ( pos );

    if ( i != _nodemap.end() ) 
    {
      v = i->second;
      return true;
    } else {
      return false;
    }
  }


  ////////////////////////////////////////////////////////////////////////////////
  template <typename value_t>
  inline void
  transferfunction<value_t>::set ( unsigned char pos, value_t const& v ) 
  {
    // overwrite old value! 
    _nodemap[pos] = v;
  }


  ////////////////////////////////////////////////////////////////////////////////
  template <typename value_t>
  inline bool 
  transferfunction<value_t>::remove ( unsigned char pos )
  {
    // do not remove if limits
    if ( pos == std::numeric::bounds<unsigned char>::highest() ||
         pos == std::numeric::bounds<unsigned char>::lowest() )
    {
      return false;
    }

    map_type::const_iterator i = _nodemap.find (pos);

    if ( i != _nodemap.end() ) {
      _nodemap.erase(i);
      return true;
    } else {
      return false;
    }
  }


  ////////////////////////////////////////////////////////////////////////////////
  template <typename value_t>
  template <typename interpolation_t>
  inline value_t
  transferfunction<value_t>::evaluate ( float v, interpolation_t intp ) const
  {
    return intp(_nodemap, v);
  }


  ////////////////////////////////////////////////////////////////////////////////
  template <typename value_t>
  template <typename container_t, typename interpolation_t>
  inline void              
  transferfunction<value_t>::evaluate ( int nsamples, container_t& container, interpolation_t interpolate) const
  {
    for (int i = 0; i != nsamples; ++i )
    {
      float t = float (i) / float (nsamples - 1);
      container.insert(container.end(), interpolate ( _nodemap, t ) );
    }
  }

} } // namespace gpucast / namespace gl
