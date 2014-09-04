/********************************************************************************
*
* Copyright (C) 2009 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : geode.cpp
*  project    : glpp
*  description:
*
********************************************************************************/

// header i/f
#include "gpucast/gl/graph/geode.hpp"

// header, system
#include <boost/foreach.hpp>

// header, project
#include <gpucast/gl/error.hpp>
#include <gpucast/gl/graph/visitor.hpp>

namespace gpucast { namespace gl {

///////////////////////////////////////////////////////////////////////////////
geode::geode()
  : node                (),
    _attributes         (),
    _indices            (new elementarraybuffer),
    _vao                (),
    _number_of_indices  (0),
    _mode               (GL_TRIANGLES),
    _material           ()
{}


///////////////////////////////////////////////////////////////////////////////
geode::~geode()
{}


///////////////////////////////////////////////////////////////////////////////
/* virtual */ void
geode::visit ( visitor const& v )
{
  v.accept(*this);
}


///////////////////////////////////////////////////////////////////////////////
void            
geode::draw ( ) const
{
  assert(_number_of_indices > 0);

  draw_elements(_mode, _number_of_indices, 0);
}


///////////////////////////////////////////////////////////////////////////////
void geode::draw_elements ( GLenum mode, std::size_t count, std::size_t start_index ) const
{
  assert(count + start_index <= _number_of_indices);

  _vao.bind();

  _indices->bind();

  // finally draw everything
  glDrawElements(mode, GLsizei(count), GL_UNSIGNED_INT, (GLvoid*)start_index);

  _vao.unbind();
}


///////////////////////////////////////////////////////////////////////////////
/* virtual */ void 
geode::compute_bbox ()
{
  // for each attribute
  BOOST_FOREACH(geode::attribute_buffer const& node, _attributes ) 
  {
    // if vertex
    if (node.type == vertex) 
    {
      // use vertices to compute axis aligned bounding box
      gpucast::math::vec4f mx = gpucast::math::vec4f::minimum();
      gpucast::math::vec4f mn = gpucast::math::vec4f::maximum();

      BOOST_FOREACH(gpucast::math::vec4f const& v, node.clientbuffer) 
      {
        mx = elementwise_max(mx, v);
        mn = elementwise_min(mn, v);

        _bbox =  gpucast::math::axis_aligned_boundingbox< gpucast::math::point3f> (  gpucast::math::point3f(mn[0], mn[1], mn[2]), 
                                                               gpucast::math::point3f(mx[0], mx[1], mx[2]));
      }
    }
  }
}


///////////////////////////////////////////////////////////////////////////////
void 
geode::add_attribute_buffer (std::size_t                location,
                             std::vector<gpucast::math::vec4f> const&  buf, 
                             attribute_type             type)
{
  // create attribute buffer
  geode::attribute_buffer attrib_buffer;

  // set client data, location for shader program and optionally type of attribute
  attrib_buffer.location      = location;
  attrib_buffer.clientbuffer  = buf;
  attrib_buffer.type          = type;

  // create new arraybuffer in server memory and copy data
  attrib_buffer.buffer        = std::shared_ptr<arraybuffer>(new arraybuffer);
  attrib_buffer.buffer->bufferdata(sizeof(gpucast::math::vec4f) * buf.size(), &buf.front());

  // store attribute buffer
  _attributes.push_back(attrib_buffer);

  // bind vertex array object
  _vao.bind();

  _vao.enable_attrib  ( location );
  _vao.attrib_array   ( *attrib_buffer.buffer, location, 4, GL_FLOAT, false, 0, 0 );

  _vao.unbind();
}


///////////////////////////////////////////////////////////////////////////////
std::size_t     
geode::attributes () const
{
  return _attributes.size();
}


///////////////////////////////////////////////////////////////////////////////
void 
geode::set_attribute_location ( std::size_t index,
                                std::size_t location)
{
  assert(index < _attributes.size());

  _attributes[index].location = location;

  // rebind attrib array to vao with new array index
  _vao.bind();

  _vao.enable_attrib  ( location );
  _vao.attrib_array   ( *_attributes[index].buffer, location, 4, GL_FLOAT, false, 0, 0 );

  _vao.unbind();
}


///////////////////////////////////////////////////////////////////////////////
std::size_t     
geode::get_attribute_location  ( std::size_t index ) const
{
  assert(index < _attributes.size());
  return _attributes[index].location;
}


///////////////////////////////////////////////////////////////////////////////
void            
geode::set_attribute_type ( std::size_t index, attribute_type type)
{
  assert(index < _attributes.size());
  _attributes[index].type = type;
}


///////////////////////////////////////////////////////////////////////////////
geode::attribute_type  
geode::get_attribute_type ( std::size_t index ) const
{
  assert(index < _attributes.size());
  return _attributes[index].type;
}


///////////////////////////////////////////////////////////////////////////////
void            
geode::set_indexbuffer ( std::vector<int> const& buf )
{
  _number_of_indices = buf.size();
  _indices->bufferdata(sizeof(int) * buf.size(), &buf.front());
}


///////////////////////////////////////////////////////////////////////////////
void            
geode::set_mode ( GLenum mode )
{
  _mode = mode;
}

///////////////////////////////////////////////////////////////////////////////
void            
geode::set_material ( material const& m )
{
  _material = m;
}

///////////////////////////////////////////////////////////////////////////////
material const& 
geode::get_material ( ) const
{
  return _material;
}

} } // namespace gpucast / namespace gl
