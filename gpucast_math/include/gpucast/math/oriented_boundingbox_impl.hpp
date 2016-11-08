/********************************************************************************
*
* Copyright (C) 2010 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : oriented_boundingbox_impl.hpp
*
*  description:
*
********************************************************************************/
#ifndef GPUCAST_MATH_ORIENTED_BOUNDING_BOX_IMPL_HPP
#define GPUCAST_MATH_ORIENTED_BOUNDING_BOX_IMPL_HPP

#include <gpucast/math/interval.hpp>

// header, system
#include <vector>
#include <limits>
#include <numeric> // std::accumulate

// header, external

// header, project
#include <gpucast/math/pow.hpp>
#include <gpucast/math/rapid.hpp>

namespace gpucast { namespace math {

  //////////////////////////////////////////////////////////////////////////////
  template <typename point_t>
  inline oriented_boundingbox<point_t>::oriented_boundingbox()
    : _center (),
      _base   (),
      _low    (),
      _high   ()
  {}


  //////////////////////////////////////////////////////////////////////////////
  template <typename point_t>
  inline oriented_boundingbox<point_t>::oriented_boundingbox ( axis_aligned_boundingbox<point_t> const& aabb )
    : _center ( (aabb.min + aabb.max) / value_type( 2) ),
      _base   ( ),
      _low    ( (aabb.max - aabb.min) / value_type(-2) ),
      _high   ( (aabb.max - aabb.min) / value_type( 2) )
  {
    assert(valid());
  }


  //////////////////////////////////////////////////////////////////////////////
  template <typename point_t>
  inline oriented_boundingbox<point_t>::oriented_boundingbox ( matrix_type const& orientation,
                                                               point_type const& center,
                                                               point_type const& low,
                                                               point_type const& high )
    : _center (center),
      _base   (orientation),
      _low    (low),
      _high   (high)
  {
    assert(valid());
  }


  //////////////////////////////////////////////////////////////////////////////
  template <typename point_t>
  template <template <typename T> class build_policy>
  inline oriented_boundingbox<point_t>::oriented_boundingbox ( pointmesh2d<point_t> const& points, build_policy<point_t> generator )
    : _center (),
      _base   (),
      _low    (),
      _high   ()
  {
    generator.apply(points);
    generator(points.begin(), points.end(), _center, _base, _low, _high);
  }


  //////////////////////////////////////////////////////////////////////////////
  template <typename point_t>
  template <template <typename T> class build_policy>
  inline oriented_boundingbox<point_t>::oriented_boundingbox ( pointmesh3d<point_t> const& points, build_policy<point_t> generator )
    : _center (),
      _base   (),
      _low    (),
      _high   ()
  {
    generator.apply(points);
    generator(points.begin(), points.end(), _center, _base, _low, _high);
  }


  //////////////////////////////////////////////////////////////////////////////
  template <typename point_t>
  template <typename iterator_t, template <typename T> class build_policy>
  inline oriented_boundingbox<point_t>::oriented_boundingbox ( iterator_t point_begin, iterator_t point_end, build_policy<point_t> generator )
    : _center (),
      _base   (),
      _low    (),
      _high   ()
  {
    generator(point_begin, point_end, _center, _base, _low, _high);
  }


  //////////////////////////////////////////////////////////////////////////////
  template <typename point_t>
  inline oriented_boundingbox<point_t>::~oriented_boundingbox()
  {}


  //////////////////////////////////////////////////////////////////////////////
  template <typename point_t>
  inline point_t
  oriented_boundingbox<point_t>::center () const
  {
    return _center;
  }


  //////////////////////////////////////////////////////////////////////////////
  template <typename point_t>
  inline typename oriented_boundingbox<point_t>::matrix_type const&
  oriented_boundingbox<point_t>::orientation () const
  {
    return _base;
  }


  //////////////////////////////////////////////////////////////////////////////
  template <typename point_t>
  inline point_t const&
  oriented_boundingbox<point_t>::low () const
  {
    return _low;
  }


  //////////////////////////////////////////////////////////////////////////////
  template <typename point_t>
  inline point_t const&
  oriented_boundingbox<point_t>::high () const
  {
    return _high;
  }


  //////////////////////////////////////////////////////////////////////////////
  template <typename point_t>
  inline typename oriented_boundingbox<point_t>::value_type
  oriented_boundingbox<point_t>::volume () const
  {
    matrix_type m;

    for (unsigned c = 0; c != point_t::coordinates; ++c)
    {
      value_type axis_length     = _high[c] - _low[c];

      for (unsigned r = 0; r != point_t::coordinates; ++r)
      {
        m[r][c] = _base[r][c] * axis_length;
      }
    }

    return fabs(m.determinant());
  }


  //////////////////////////////////////////////////////////////////////////////
  template <typename point_t>
  inline typename oriented_boundingbox<point_t>::value_type
  oriented_boundingbox<point_t>::surface () const
  {
    value_type surf_total = 0;

    for (unsigned i = 0; i != point_t::coordinates; ++i)
    {
      for (unsigned j = i; j != point_t::coordinates; ++j)
      {
        if (i != j) {
          surf_total += 2 * (_high[i] - _low[i]) * (_high[j] - _low[j]);
        }
      }
    }

    return surf_total;
  }


  //////////////////////////////////////////////////////////////////////////////
  template <typename point_t>
  inline bool
  oriented_boundingbox<point_t>::is_inside( point_t const& p ) const
  {
    point_t p_local = _base.inverse() * (p - _center);
    bool inside = true;

    for (unsigned int i = 0; i != point_t::coordinates; ++i)
    {
      inside &= (_low[i] <= p_local[i]) && (_high[i] >= p_local[i]);
    }

    return inside;
  }


  //////////////////////////////////////////////////////////////////////////////
  template <typename point_t>
  inline bool
  oriented_boundingbox<point_t>::is_inside ( oriented_boundingbox<point_t> const& other ) const
  {
    throw std::runtime_error("not implemented yet");
  }


  //////////////////////////////////////////////////////////////////////////////
  template <typename point_t>
  inline bool
  oriented_boundingbox<point_t>::overlap ( oriented_boundingbox<point_t> const& a ) const
  {
    if (N == 3) {
      return overlap3d(*this, a);
    } else {
      throw std::runtime_error("not implemented yet");
    }
  }


  //////////////////////////////////////////////////////////////////////////////
  template <typename point_t>
  bool
  oriented_boundingbox<point_t>::overlap ( axis_aligned_boundingbox<point_t> const& a ) const
  {
    if (N == 3)
    {
      oriented_boundingbox<point_t> tmp ( a );
      return overlap3d(*this, tmp);
    } else {
      throw std::runtime_error("not implemented yet");
    }
  }


  //////////////////////////////////////////////////////////////////////////////
  template <typename point_t>
  template <typename insert_iterator>
  inline void
  oriented_boundingbox<point_t>::generate_corners ( insert_iterator ins ) const
  {
    std::size_t vertices =  gpucast::math::meta_pow<2, point_t::coordinates>::result;

    for (std::size_t v = 0; v != vertices; ++v)
    {
      point_t corner;
      std::size_t minmax_bitmask = v;

      for (std::size_t i = 0; i != point_t::coordinates; ++i)
      {
        corner[i] = (minmax_bitmask & 0x01) ? _high[i] : _low[i];
        minmax_bitmask >>= 1;
      }

      ins = _center + _base * corner;
    }
  }


  //////////////////////////////////////////////////////////////////////////////
  template <typename point_t>
  void
  oriented_boundingbox<point_t>::generate_corners ( std::vector<point_t>& result ) const
  {
    generate_corners(std::back_inserter(result));
  }


  //////////////////////////////////////////////////////////////////////////////
  template <typename point_t>
  bool
  oriented_boundingbox<point_t>::valid () const
  {
    for (auto i = 0; i != point_t::coordinates; ++i ) {
      _low[i] < _high[i];
    }
    return _base.valid() && _low.valid() && _high.valid() && _center.valid();
  }

  //////////////////////////////////////////////////////////////////////////////
  template <typename point_t>
  axis_aligned_boundingbox<point_t>
  oriented_boundingbox<point_t>::aabb () const
  {
    std::vector<point_t> corners;
    generate_corners(std::back_inserter(corners));
    return axis_aligned_boundingbox<point_t> (corners.begin(), corners.end());
  }


  //////////////////////////////////////////////////////////////////////////////
  template<typename point_t>
  void
  oriented_boundingbox<point_t>::uniform_split ( std::vector<typename abstract_boundingbox<point_t>::pointer_type>& l ) const
  {
    throw std::runtime_error("not implemented yet");
  }


  //////////////////////////////////////////////////////////////////////////////
  template <typename point_t>
  inline /* virtual */ void
  oriented_boundingbox<point_t>::print(std::ostream& os) const
  {
    std::cout << "center       : " << _center << std::endl;
    std::cout << "base vectors : " << std::endl << _base   << std::endl;
    std::cout << "low          : " << _low    << std::endl;
    std::cout << "high         : " << _high   << std::endl;
  }


  //////////////////////////////////////////////////////////////////////////////
  template <typename point_t>
  /* virtual */ void          
  oriented_boundingbox<point_t>::write ( std::ostream& os ) const
  {
    os.write(reinterpret_cast<char const*>(&_center[0]), sizeof(point_type));
   _base.write(os);
   os.write(reinterpret_cast<char const*>(&_low[0]), sizeof(point_type));
   os.write(reinterpret_cast<char const*>(&_high[0]), sizeof(point_type));
  }


  //////////////////////////////////////////////////////////////////////////////
  template <typename point_t>
  /* virtual */ void          
  oriented_boundingbox<point_t>::read ( std::istream& is )
  {
    is.read(reinterpret_cast<char*>(&_center[0]), sizeof(point_type));
    _base.read(is);
    is.read(reinterpret_cast<char*>(&_low[0]), sizeof(point_type));
    is.read(reinterpret_cast<char*>(&_high[0]), sizeof(point_type));
  }


  //////////////////////////////////////////////////////////////////////////////
  template <typename point_t>
  inline std::ostream& operator<<(std::ostream& os, oriented_boundingbox<point_t> const& a)
  {
    a.print(os);
    return os;
  }

  //////////////////////////////////////////////////////////////////////////////
  template <typename point_t>
  template <typename value_t>
  std::vector<math::vec4<value_t>> oriented_boundingbox<point_t>::serialize() const
  {
    std::vector<math::vec4<value_t>> result;

    auto pcenter = math::vec4<value_t>(center()[0], center()[1], center()[2], 1.0);
    auto phigh = math::vec4<value_t>(high()[0], high()[1], high()[2], 0.0f);
    auto plow = math::vec4<value_t>(low()[0], low()[1], low()[2], 0.0f);

    result.push_back(pcenter);
    result.push_back(phigh);
    result.push_back(plow);

    auto orientation_mat = orientation();
    auto inv_orientation_mat = compute_inverse(orientation_mat);

    result.push_back(math::vec4<value_t>(orientation_mat[0][0], orientation_mat[1][0], orientation_mat[2][0], 0.0));
    result.push_back(math::vec4<value_t>(orientation_mat[0][1], orientation_mat[1][1], orientation_mat[2][1], 0.0));
    result.push_back(math::vec4<value_t>(orientation_mat[0][2], orientation_mat[1][2], orientation_mat[2][2], 0.0));
    result.push_back(math::vec4<value_t>(0.0, 0.0, 0.0, 1.0));

    result.push_back(math::vec4<value_t>(inv_orientation_mat[0][0], inv_orientation_mat[1][0], inv_orientation_mat[2][0], 0.0));
    result.push_back(math::vec4<value_t>(inv_orientation_mat[0][1], inv_orientation_mat[1][1], inv_orientation_mat[2][1], 0.0));
    result.push_back(math::vec4<value_t>(inv_orientation_mat[0][2], inv_orientation_mat[1][2], inv_orientation_mat[2][2], 0.0));
    result.push_back(math::vec4<value_t>(0.0, 0.0, 0.0, 1.0));

    auto lbf = point_type(plow[0], plow[1], plow[2], 1.0);  // left, bottom, front
    auto rbf = point_type(phigh[0], plow[1], plow[2], 1.0);  // right, bottom, front
    auto rtf = point_type(phigh[0], phigh[1], plow[2], 1.0);  // right, top, front
    auto ltf = point_type(plow[0], phigh[1], plow[2], 1.0);  // left, top, front

    auto lbb = point_type(plow[0], plow[1], phigh[2], 1.0); // left, bottom, back  
    auto rbb = point_type(phigh[0], plow[1], phigh[2], 1.0); // right, bottom, back  
    auto rtb = point_type(phigh[0], phigh[1], phigh[2], 1.0); // right, top, back  
    auto ltb = point_type(plow[0], phigh[1], phigh[2], 1.0); // left, top, back  

    //lbf = orientation_mat * lbf;
    //rbf = orientation_mat * rbf;
    //rtf = orientation_mat * rtf;
    //ltf = orientation_mat * ltf;
    //                           
    //lbb = orientation_mat * lbb;
    //rbb = orientation_mat * rbb;
    //rtb = orientation_mat * rtb;
    //ltb = orientation_mat * ltb;

    lbf.weight(1.0);
    rbf.weight(1.0);
    rtf.weight(1.0);
    ltf.weight(1.0);

    result.push_back(math::vec4<value_t>(lbf));
    result.push_back(math::vec4<value_t>(rbf));
    result.push_back(math::vec4<value_t>(rtf));
    result.push_back(math::vec4<value_t>(ltf));

    result.push_back(math::vec4<value_t>(lbb));
    result.push_back(math::vec4<value_t>(rbb));
    result.push_back(math::vec4<value_t>(rtb));
    result.push_back(math::vec4<value_t>(ltb));

    return result;
  }

#if 1

  //////////////////////////////////////////////////////////////////////////////
  template <typename point_t>
  bool overlap3d (oriented_boundingbox<point_t> const& A, oriented_boundingbox<point_t> const& B)
  {
    // Gottschalk et al. : OBBTrees and RAPID
    // transform OBB's that lower left corner of A lies in origin
    // B's size and orientation are given in coordinate system of A
    //typename oriented_boundingbox<point_t>::matrix_type R = B.orientation().inverse() * A.orientation();
    typename oriented_boundingbox<point_t>::matrix_type R = B.orientation() * A.orientation().inverse();
    point_t d = B.center() - A.center();
    point_t T = A.orientation().inverse() * d;

    // OBB assumed to be symmetric -> abs(low) == abs(high) -> size equals high()
    point_t a = A.high();
    point_t b = B.high();
    //point_t b = R * B.high();

    //R = R.transpose();

    int result = obb_disjoint<typename point_t::value_type> ( R, T, a, b);

    //return (result == 0) || B.is_inside(A.center()) || A.is_inside(B.center());
    //return B.is_inside(A.center()) || A.is_inside(B.center());
    return (result == 0);
  }

#else
  //////////////////////////////////////////////////////////////////////////////
  template <typename point_t>
  bool overlap3d (oriented_boundingbox<point_t> const& A, oriented_boundingbox<point_t> const& B)
  {
    // brute force! -> transform into coordinate system of A -> A becomes AABB
    // do is_inside test and edge-AABB-intersections with all edges of B
    typename oriented_boundingbox<point_t>::matrix_type R = B.orientation() * A.orientation().inverse();
    point_t d = B.center() - A.center();
    point_t T = A.orientation().inverse() * d;

    // get obb's base vectors to determine ray-aabb intersections
    point3d dir1(R.col(0));
    point3d dir2(R.col(1));
    point3d dir3(R.col(2));

    dir1 /= dir1.length();
    dir2 /= dir2.length();
    dir3 /= dir3.length();

    // 1. fast inside test with center
    bool center_inside = B.is_inside(A.center()) || A.is_inside(B.center());

    if ( center_inside ) {
      return true;
    }

    // 2. continue with edge bbox intersections
    std::vector<point3d> origins;

    // generate vertices of B
    B.generate_corners(std::back_inserter(origins));

    // transform vertices into coordinate system of A
    std::for_each(origins.begin(), origins.end(), [&]( point3d& p ) { p = (R * p) - d;} );




    return true;
  }
#endif


} } // namespace gpucast / namespace math

#endif // GPUCAST_MATH_ORIENTED_BOUNDING_BOX_IMPL_HPP

