/********************************************************************************
*
* Copyright (C) 2010 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : prefilter2d.hpp
*
*  description:
*
********************************************************************************/
#ifndef GPUCAST_MATH_PREFILTER2D_HPP
#define GPUCAST_MATH_PREFILTER2D_HPP

//#include <algorithm>
#include <numeric>
#include <vector>

#include <gpucast/math/util/gauss2d.hpp>
#include <gpucast/math/util/discretizer2d.hpp>
#include <gpucast/math/util/signed_gradient.hpp>

namespace gpucast { namespace math {
  namespace util {

  template <typename vec2_t>
  class prefilter2d
  {
  public :

    typedef typename vec2_t::value_type value_type;

  public :

    ///////////////////////////////////////////////////////////////////////////
    prefilter2d(size_t samples = 64, value_type gauss_sigma = 0.5)
      : _nsamples   ( samples ),
        _sigma      ( gauss_sigma ),
        _samples    ( samples*samples ),
        _weights    ( samples*samples ),
        _maxintegral( 0 )
    {
      discretizer2d<vec2_t> dtizer(_nsamples, _nsamples, value_type(-0.5), value_type(0.5), value_type(-0.5), value_type(0.5));

      std::generate (_samples.begin(), _samples.end(), dtizer);
      std::transform(_samples.begin(), _samples.end(), _weights.begin(), gauss2d<vec2_t>(_sigma));

      _maxintegral = std::accumulate(_weights.begin(), _weights.end(), value_type(0));
    }


    ///////////////////////////////////////////////////////////////////////////
    value_type
    operator()( vec2_t const& angle_radius) const
    {
      if (angle_radius[1] == 0) {
        return 0.5;
      }
      else {
        value_type integral(0);
        classify_sample_by_signed_gradient<vec2_t> insidetest;

        typename std::vector<value_type>::const_iterator w = _weights.begin();
        for (typename std::vector<vec2_t>::const_iterator s = _samples.begin(); s != _samples.end(); ++s, ++w)
        {
          bool is_inside = insidetest(angle_radius, *s);
          integral += (*w) * value_type(is_inside);
        }

        return integral / _maxintegral;
      }
    }

  private :

    size_t                  _nsamples;
    value_type              _sigma;
    std::vector<vec2_t>     _samples;
    std::vector<value_type> _weights;
    value_type              _maxintegral;
  };

  } // namespace util
} } // namespace gpucast / namespace math

#endif // GPUCAST_MATH_PREFILTER2D_HPP
