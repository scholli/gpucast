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
#include <gpucast/math/parametric/domain/partition/monotonic_contour/kdsplit_mid.hpp>
#include <gpucast/math/parametric/domain/partition/monotonic_contour/kdsplit_greedy.hpp>
#include <gpucast/math/parametric/domain/partition/monotonic_contour/kdsplit_maxarea.hpp>
#include <gpucast/math/parametric/domain/partition/monotonic_contour/kdsplit_sah.hpp>
#include <gpucast/math/parametric/domain/partition/monotonic_contour/kdsplit_minsplit.hpp>

namespace gpucast {
  namespace math {
    namespace domain {

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
template <typename value_t>
contour_map_kd<value_t>::contour_map_kd(kd_split_strategy s, bool usebitfield, unsigned resolution)
  : _split_strategy(s),
    _enable_bitfield(usebitfield),
    _classification_texture(resolution, resolution)
{}

///////////////////////////////////////////////////////////////////////////////
template <typename value_t>
bool
contour_map_kd<value_t>::initialize()
{
  this->minimize_overlaps();

  std::set<contour_segment_ptr> segment_set(this->_contour_segments.begin(), this->_contour_segments.end());
  bool success = false;

  switch (_split_strategy)
  {
    case midsplit: {
      success = _kdtree.initialize(kdsplit_mid<value_t>(), segment_set);
      break;
      }
    case greedy: {
      success = _kdtree.initialize(kdsplit_greedy<value_t>(), segment_set);
      break;
      }
    case maxarea: {
      success = _kdtree.initialize(kdsplit_maxarea<value_t>(), segment_set);
      break;
    }
    case sah: {
      success = _kdtree.initialize(kdsplit_sah<value_t>(), segment_set);
      break;
    }
    case minsplits : {
      success = _kdtree.initialize(kdsplit_minsplit<value_t>(), segment_set);
      break;
    }
    default: 
      return false;
  };

  // create bitfield
  if (_enable_bitfield) {
    value_type texel_size_u = _kdtree.bbox.size()[point_type::u] / _classification_texture.width();
    value_type texel_size_v = _kdtree.bbox.size()[point_type::v] / _classification_texture.height();

    for (std::size_t y = 0; y != _classification_texture.height(); ++y) {
      for (std::size_t x = 0; x != _classification_texture.width(); ++x) {
        bbox_type texel(_kdtree.bbox.min + point_type(texel_size_u * (x), texel_size_v * (y)),
                        _kdtree.bbox.min + point_type(texel_size_u * (1 + x), texel_size_v * (1 + y)));
        auto node = _kdtree.is_in_node(texel);
        if (node) {
          if (node->overlapping_segments.empty()) {
            // use parity for classification
            _classification_texture(x, y) = 1 + node->parity;
          }
          else {
            // not classifyable
            _classification_texture(x, y) = 0;
          }
        }
        else {
          // not classifyable
          _classification_texture(x, y) = 0;
        }
      }
    }
  }

#if 0
  std::vector<kdnode_ptr> nodes;
  _kdtree.root->serialize_dfs(nodes);
  for (auto node : nodes) {
    if (node->is_leaf()) {
      std::cout << "depth : " << node->depth << " , area : " << node->bbox.size().abs() << std::endl;
    }
  }

  std::cout << "KD tree build successfully. Traversal costs : " << _kdtree.traversal_costs() << std::endl;
#endif
  return success;
}

///////////////////////////////////////////////////////////////////////////////
template <typename value_t>
kdtree2d<value_t> const& contour_map_kd<value_t>::partition() const 
{
  return _kdtree;
}

///////////////////////////////////////////////////////////////////////////////
template <typename value_t>
classification_field<unsigned char> const& contour_map_kd<value_t>::pre_classification() const
{
  return _classification_texture;
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
std::ostream& operator<<(std::ostream& os,  contour_map_kd<value_t> const& rhs)
{
  // reverse, if not increasing in v-direction
  for (auto const& c : rhs.monotonic_segments()) {
    os << c << std::endl;
  }

  return os;
}

    } // namespace domain
  } // namespace math
} // namespace gpucast 
