/********************************************************************************
*
* Copyright (C) 2007-2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : convex_hull.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_CORE_CONVEX_HULL_HPP
#define GPUCAST_CORE_CONVEX_HULL_HPP

#ifdef _MSC_VER
 #pragma warning(disable: 4251)
#endif

// header, system
#include <vector>

// header, project
#include <gpucast/core/gpucast.hpp>

#include <gpucast/math/vec3.hpp>
#include <gpucast/math/vec4.hpp>




namespace gpucast {

////////////////////////////////////////////////////////////////////////////////
// triangle print helper ///////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
template <typename point_t>
class triangle_printer {
public :
  triangle_printer(std::vector<point_t> const& points, std::ostream& os)
    : points_(points),
      os_(os)
  {}

  void
  operator()(int const& tri) const {
    os_ << "Vertex : " << points_.at(tri) << std::endl;
  }

  std::vector<point_t> const& points_;
  std::ostream& os_;
};


class GPUCAST_CORE convex_hull 
{
public : // enums, typedefs

  typedef std::vector<int>::const_iterator            const_index_iterator;

  typedef std::vector<gpucast::math::vec3d>::iterator          vec3d_iterator;
  typedef std::vector<gpucast::math::vec3d>::const_iterator    const_vec3d_iterator;

  typedef std::vector<gpucast::math::vec4f>::iterator          vec4f_iterator;
  typedef std::vector<gpucast::math::vec4f>::const_iterator    const_vec4f_iterator;

public : // c'tors / d'tor

  convex_hull();
  convex_hull(convex_hull const&);
  ~convex_hull();

  convex_hull& operator=(convex_hull const&);

public : // methods

  //////////////////////////////////////////////////////////////////////////////
  void          set           ( vec3d_iterator       vertex_begin,
                                vec3d_iterator       vertex_end,
                                const_vec4f_iterator texcoord_begin,
                                const_vec4f_iterator texcoord_end);

  //////////////////////////////////////////////////////////////////////////////
  void          swap          ( convex_hull& );

  std::size_t   size          () const;
  void          clear         ();

  void          print         ( std::ostream& ) const;
  void          merge         ( convex_hull const& );

  //////////////////////////////////////////////////////////////////////////////
  void          set_trimid    ( std::size_t id );
  void          set_dataid    ( std::size_t id );
  void          set_parameter ( gpucast::math::vec4d const& parameter_range );
  void          set_trimtype  ( bool outer_trim );
  void          set_order     ( std::size_t u, std::size_t v );

  //////////////////////////////////////////////////////////////////////////////
  const_vec3d_iterator vertices_begin () const;
  const_vec3d_iterator vertices_end   () const;

  const_vec4f_iterator uv_begin       () const;
  const_vec4f_iterator uv_end         () const;

  const_index_iterator indices_begin  () const;
  const_index_iterator indices_end    () const;

private : // data

  // vertices necessary for convex hull and their texcoords
  std::vector<gpucast::math::vec3d>  _vertices;
  std::vector<gpucast::math::vec4f>  _parameter;

  // resulting triangle id's
  std::vector<int>          _indices;
};

  std::ostream& operator<<(std::ostream& os, convex_hull const& rhs);

} // namespace gpucast

#endif // GPUCAST_CORE_CONVEX_HULL_HPP
