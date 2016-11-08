/********************************************************************************
*
* Copyright (C) 2010 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : oriented_boundingbox.hpp
*
*  description:
*
********************************************************************************/
#ifndef GPUCAST_MATH_ORIENTED_BOUNDING_BOX_HPP
#define GPUCAST_MATH_ORIENTED_BOUNDING_BOX_HPP

// header, system
#include <vector>

// header, project
#include <gpucast/math/vec4.hpp>
#include <gpucast/math/parametric/point.hpp>
#include <gpucast/math/parametric/pointmesh2d.hpp>
#include <gpucast/math/parametric/pointmesh3d.hpp>

#include <gpucast/math/matrix.hpp>
#include <gpucast/math/abstract_boundingbox.hpp>
#include <gpucast/math/axis_aligned_boundingbox.hpp>


namespace gpucast { namespace math {

  template <typename point_t>
  class oriented_boundingbox : public abstract_boundingbox<point_t>
  {
  public : // typedefs

    static unsigned const N = point_t::coordinates;

    typedef oriented_boundingbox<point_t> type;
    typedef std::shared_ptr<type>         pointer_type;

    typedef typename point_t::value_type  value_type;
    typedef point_t                       point_type;
    typedef matrix<value_type, N, N>      matrix_type;

  public : // ctor/dtor

    oriented_boundingbox              ();

    oriented_boundingbox              ( axis_aligned_boundingbox<point_t> const& );

    oriented_boundingbox              ( matrix_type const& orientation,
                                        point_type const& center,
                                        point_type const& low,
                                        point_type const& high );

    template <template <typename T> class build_policy>
    oriented_boundingbox              ( pointmesh2d<point_t> const& points,
                                        build_policy<point_type> policy );

    template <template <typename T> class build_policy>
    oriented_boundingbox              ( pointmesh3d<point_t> const& points,
                                        build_policy<point_type> policy );

    template <typename iterator_t, template <typename T> class build_policy>
    oriented_boundingbox              ( iterator_t point_begin,
                                        iterator_t point_end,
                                        build_policy<point_type> policy );

    virtual ~oriented_boundingbox     ();

  public : // methods

    point_t               center            () const;
    matrix_type const&    orientation       () const;
    point_t const&        low               () const;
    point_t const&        high              () const;

    value_type            volume            () const;
    value_type            surface           () const;

    bool                  is_inside         ( point_t const& p ) const;
    bool                  is_inside         ( oriented_boundingbox<point_t> const& a ) const;

    bool                  overlap           ( oriented_boundingbox<point_t> const& a ) const;
    bool                  overlap           ( axis_aligned_boundingbox<point_t> const& a ) const;

    template <typename insert_iterator>
    void                  generate_corners  ( insert_iterator i ) const;
    void                  generate_corners  ( std::vector<point_t>& ) const;

    bool                  valid             () const;

    axis_aligned_boundingbox<point_t> aabb  ( ) const;

    virtual void          uniform_split     ( std::vector<typename abstract_boundingbox<point_t>::pointer_type>& l ) const;

    virtual void          print             ( std::ostream& os ) const;
    virtual void          write             ( std::ostream& os ) const;
    virtual void          read              ( std::istream& is );

    template <typename value_t>
    std::vector<math::vec4<value_t>> serialize() const;

  protected : // attributes

    point_t                _center;
    matrix_type            _base;
    point_t                _low;
    point_t                _high;
  };

  template <typename point_t>
  std::ostream& operator<<(std::ostream& os, oriented_boundingbox<point_t> const& a);

  template <typename point_t>
  bool overlap3d (oriented_boundingbox<point_t> const& a, oriented_boundingbox<point_t> const& b);

  typedef oriented_boundingbox<point2f> obbox2f;
  typedef oriented_boundingbox<point3f> obbox3f;
  typedef oriented_boundingbox<point2d> obbox2d;
  typedef oriented_boundingbox<point3d> obbox3d;

} } // namespace gpucast / namespace math

#include <gpucast/math/oriented_boundingbox_impl.hpp>

#endif // GPUCAST_MATH_ORIENTED_BOUNDING_BOX_HPP

