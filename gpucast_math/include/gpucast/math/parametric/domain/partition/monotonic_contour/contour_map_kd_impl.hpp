/********************************************************************************
*
* Copyright (C) 2013 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : contour_map_kd_impl.hpp
*
*  description:
*
********************************************************************************/
// includes, system

// includes, project

namespace gpucast {
  namespace math {
    namespace domain {

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
template <typename value_t>
void
contour_map_kd<value_t>::initialize()
{
}

///////////////////////////////////////////////////////////////////////////////
template <typename value_t>
void
contour_map_kd<value_t>::print(std::ostream& os) const
{
  os << "contour_map_kd<value_t>::print() not implemented" << std::endl;
}

///////////////////////////////////////////////////////////////////////////////
template <typename value_t>
std::ostream& operator<<(std::ostream& os,  gpucast::math::contour_map_kd<value_t> const& rhs)
{
  return os;
}

    } // namespace domain
  } // namespace math
} // namespace gpucast 
