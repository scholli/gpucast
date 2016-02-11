/********************************************************************************
*
* Copyright (C) 2015 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : classification_field.hpp
*
*  description:
*
********************************************************************************/
#ifndef GPUCAST_MATH_CLASSICATION_FIELD_HPP
#define GPUCAST_MATH_CLASSICATION_FIELD_HPP

// includes, system


namespace gpucast {
  namespace math {
    namespace domain {

/////////////////////////////////////////////////////////////////////////////
// predicate class, e.g. for copy_if operations
/////////////////////////////////////////////////////////////////////////////
template <typename value_t>
class classification_field
{
  public : // typedef

    classification_field(std::size_t width, std::size_t height) 
      : _width(width),
        _height(height),
        _data(width*height)
    {}

    classification_field(std::size_t width, std::size_t height, value_t* data)
      : _width(width),
        _height(height), 
        _data(width*height)
    {
      std::copy(data, data + width*height, _data.begin())
    }

  public : // c'tor

    value_t& operator()(std::size_t x, std::size_t y) {
      return _data[y*_width + x];
    }

    value_t const& operator()(std::size_t x, std::size_t y) const {
      return _data[y*_width + x];
    }

    value_t const* data() const {
      return &_data[0];
    }

    std::size_t height() const {
      return _height;
    }

    std::size_t width() const {
      return _width;
    }

  private : // attributes

    std::size_t          _width;
    std::size_t          _height;
    std::vector<value_t> _data;
};

    } // namespace domain
  } // namespace math
} // namespace gpucast 

#endif // GPUCAST_MATH_CLASSICATION_FIELD_HPP
