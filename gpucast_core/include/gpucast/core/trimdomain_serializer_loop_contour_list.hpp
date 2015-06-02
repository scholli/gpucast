/********************************************************************************
*
* Copyright (C) 2013 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : trimdomain_serializer_loop_contour_list.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_CORE_TRIMDOMAIN_LOOP_CONTOUR_LIST_HPP
#define GPUCAST_CORE_TRIMDOMAIN_LOOP_CONTOUR_LIST_HPP

// header, system
#include <gpucast/math/parametric/domain/partition/monotonic_contour/contour_map_loop_list.hpp>

// header, project
#include <gpucast/core/gpucast.hpp>
#include <gpucast/core/conversion.hpp>

#include <gpucast/core/trimdomain.hpp>
#include <gpucast/core/trimdomain_serializer.hpp>

namespace gpucast {

class trimdomain_serializer_loop_contour_list : public trimdomain_serializer
{
  public : // enums/typedefs

    using trimdomain_serializer::address_type;

    typedef trimdomain::value_type                                   value_type;
    typedef gpucast::math::domain::contour_map_loop_list<value_type> partition_type;
    typedef partition_type::contour_segment_ptr                      contour_segment_ptr;
    typedef partition_type::contour_ptr                              contour_ptr;

    struct serialization 
    {
      serialization() 
      : loops(1),
        contours(1),
        curves(1),
        points(1)
      {}

      struct bbox_t {
        float umin; 
        float umax;
        float vmin;
        float vmax;
      };

      struct point_t { 
        float wx;
        float wy; 
        float w;
        float pad;
      };

      struct curve_t { 
        unsigned order;
        unsigned point_index;
        unsigned uincreasing;
        unsigned pad;
        bbox_t   bbox;
      };

      struct loop_t {
        unsigned nchildren;
        unsigned child_index;
        unsigned ncontours;
        unsigned contour_index;
        bbox_t   bbox;
      };

      struct contour_t {
        unsigned ncurves;
        unsigned curve_index;
        unsigned uincreasing;
        unsigned parity_priority;
        bbox_t   bbox;
      };

      std::vector<loop_t>    loops;
      std::vector<contour_t> contours;
      std::vector<curve_t>   curves;
      std::vector<point_t>   points;
    };

  public : // methods

    address_type     serialize  ( trimdomain_ptr const&                               input, 
                                  std::unordered_map<trimdomain_ptr, address_type>&   referenced_trimdomains,
                                  serialization& result) const;

    void             serialize  ( contour_segment_ptr const& input, partition_type const& partition, serialization& result) const;

    void             serialize  ( curve_ptr const& curve, serialization& result ) const;

    address_type     serialize  ( partition_type::trimloop const& loop, partition_type const& partition, serialization& result, bool is_outer_loop) const;

  private : // member

};

std::ostream& operator<<(std::ostream& os, trimdomain_serializer_loop_contour_list::serialization::point_t const& p) {
  os << "point = [" << p.wx << ", " << p.wy << ", " << p.w << "]";
  return os;
}

std::ostream& operator<<(std::ostream& os, trimdomain_serializer_loop_contour_list::serialization::bbox_t const& p) {
  os << "bbox = [" << p.umin << ", " << p.umax << ", " << p.vmin << ", " << p.vmax << "]";
  return os;
}

std::ostream& operator<<(std::ostream& os, trimdomain_serializer_loop_contour_list::serialization::curve_t const& p) {
  os << "curve = [order=" << p.order << ", point_index=" << p.point_index << ", u_increasing=" << p.uincreasing << "], " << p.bbox;
  return os;
}

std::ostream& operator<<(std::ostream& os, trimdomain_serializer_loop_contour_list::serialization::loop_t const& p) {
  os << "loop = [child_index" << p.child_index << ", children=" << p.nchildren << ", ncontours=" << p.ncontours << ", contour_index=" << p.contour_index << "], " << p.bbox;
  return os;
}

std::ostream& operator<<(std::ostream& os, trimdomain_serializer_loop_contour_list::serialization::contour_t const& p) {
  auto parity_priority = uintToUint2x16(p.parity_priority);
  os << "contour = [curve_index=" << p.curve_index << ", ncurves=" << p.ncurves << ", parity=" << parity_priority[0] << ", priority=" << parity_priority[1] << ", u_increasing=" << p.uincreasing << "]" << "], " << p.bbox;
  return os;
}

/////////////////////////////////////////////////////////////////////////////
trimdomain_serializer::address_type  
trimdomain_serializer_loop_contour_list::serialize(trimdomain_ptr const&                                    input_domain,
                                                   std::unordered_map<trimdomain_ptr, address_type>&        referenced_trimdomains,
                                                   trimdomain_serializer_loop_contour_list::serialization&  result) const
{
  // if already in buffer -> return index
  if (referenced_trimdomains.count(input_domain)) {
    return referenced_trimdomains.find(input_domain)->second;
  }

  partition_type loop_list;

  for (auto const& loop : input_domain->loops()) {
    loop_list.add(loop);
  }

  loop_list.initialize();

  // not trimmed 
  if (loop_list.loops().empty())
  {
    auto loop_index = loop_list.loops().size();

    serialization::loop_t no_loop{ 0, 0, 0, 0, { 0.0f, 0.0f, 0.0f, 0.0f } };
    result.loops.push_back(no_loop);

    referenced_trimdomains.insert(std::make_pair(input_domain, loop_index));
    return loop_index;
  }
  else {

    address_type root_index = serialize(loop_list.root(), loop_list, result, true);
    return root_index;
    /*
    serialization::loop_t parent_loop{ loop_list.loops().size() - 1,
      result.loops.size() + 1,
      0, 0,
      {
      loop_list.loops().front().bbox().min[0],
      loop_list.loops().front().bbox().max[0],
      loop_list.loops().front().bbox().min[1],
      loop_list.loops().front().bbox().max[1] }
  }; // TODO
    result.loops.push_back(parent_loop);

    // add children
    for (unsigned i = 1; i != loop_list.loops().size(); ++i)
    {
      serialization::loop_t child_loop{ 0, 0, 0, 0, { 0.0f, 0.0f, 0.0f, 0.0f } };
      result.loops.push_back(parent_loop);
    }
    */
  }


}

/////////////////////////////////////////////////////////////////////////////
void
trimdomain_serializer_loop_contour_list::serialize(contour_segment_ptr const& contour, partition_type const& loop_list, serialization& result) const
{
  auto contour_bbox = contour->bbox();
  auto curve_index = result.curves.size();

  for (auto i = contour->begin(); i != contour->end(); ++i) {
    serialize(*i, result);
  }

  unsigned int parity_priority = uint2x16ToUInt(loop_list.parity(contour), loop_list.priority(contour));

  serialization::contour_t serialized_contour{ contour->size(), 
                                               curve_index,
                                               contour->is_increasing(point_type::u), 
                                               parity_priority,
                                               { contour_bbox.min[point_type::u],
                                                 contour_bbox.max[point_type::u],
                                                 contour_bbox.min[point_type::v],
                                                 contour_bbox.max[point_type::v] }
  };

  result.contours.push_back(serialized_contour);
}

/////////////////////////////////////////////////////////////////////////////
void             
trimdomain_serializer_loop_contour_list::serialize(curve_ptr const& curve, serialization& result) const
{
  bbox_type curve_bbox;
  curve->bbox_simple(curve_bbox);

  auto point_index = result.points.size();

  serialization::curve_t serialized_curve{ 
    curve->order(), 
    point_index, 
    curve->is_increasing(point_type::u),
    0,
    {
    curve_bbox.min[point_type::u],
    curve_bbox.max[point_type::u],
    curve_bbox.min[point_type::v],
    curve_bbox.max[point_type::v]
    }
  };

  for (auto p = curve->begin(); p != curve->end(); ++p) {
    serialization::point_t serialized_point = { (*p)[0] * p->weight(), 
                                                (*p)[1] * p->weight(),
                                                p->weight(), 
                                                0.0 };
    result.points.push_back(serialized_point);
  }

  result.curves.push_back(serialized_curve);
}





/////////////////////////////////////////////////////////////////////////////
trimdomain_serializer::address_type     
trimdomain_serializer_loop_contour_list::serialize(partition_type::trimloop const& loop, partition_type const& loop_list, serialization& result, bool is_outer_loop) const
{
  // get index and allocate memory for loop and children loops
  auto nchildren         = loop.children.size();
  bool child_index_set   = false;
  auto first_child_index = 0;

  // serialize children
  for (auto const& child_loop : loop.children) {
    auto child_index = serialize(child_loop, loop_list, result, false);
    if (!child_index_set) {
      first_child_index = child_index;
      child_index_set = true;
    }
  }

  auto ncontours = loop.contour->monotonic_segments().size();
  auto contour_index = result.contours.size();

  for (auto const& contour : loop.contour->monotonic_segments()) {
    serialize(contour, loop_list, result);
  }

  // for now, only support for depth 2, alternatively use BFS-traversal
  assert(loop.children.empty() || is_outer_loop);

  auto loop_bbox = loop.contour->bbox();
  serialization::loop_t serialized_loop{ nchildren,
                                         first_child_index,
                                         ncontours,
                                         contour_index,
                                          { loop_bbox.min[point_type::u], 
                                            loop_bbox.max[point_type::u], 
                                            loop_bbox.min[point_type::v], 
                                            loop_bbox.max[point_type::v] }
  };
                               
  auto loop_index = result.loops.size();
  result.loops.push_back(serialized_loop);
    
  return loop_index;
}

} // namespace gpucast

#endif // GPUCAST_CORE_TRIMDOMAIN_LOOP_CONTOUR_LIST_HPP
