/********************************************************************************
*
* Copyright (C) 2007-2011 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : serialize_tree_dfs_traversal.cpp
*  project    : gpucast
*  description:
*
********************************************************************************/

// header i/f
#include "gpucast/volume/isosurface/octree/serialize_tree_dfs_traversal.hpp"

// header, system

// header, external

// header, project
#include <gpucast/volume/isosurface/octree/ocnode.hpp>

namespace gpucast {

///////////////////////////////////////////////////////////////////////////////
serialize_tree_dfs_traversal::serialize_tree_dfs_traversal()
  : nodevisitor     (),
    ocnode_buffer   ( new std::vector<gpucast::gl::vec4u>(1) ), 
    facelist_buffer ( new std::vector<gpucast::gl::vec4u>(1) ), 
    bbox_buffer     ( new std::vector<gpucast::gl::vec4f>(1) ), 
    range_buffer    ( new std::vector<float>(1) ),
    placeholder     ( new std::vector<std::size_t> ),
    ocnode_map      ( new std::map<std::size_t, unsigned> ),
    face_map        ( new std::map<face_ptr, gpucast::gl::vec4u> )
{}


///////////////////////////////////////////////////////////////////////////////
serialize_tree_dfs_traversal::~serialize_tree_dfs_traversal()
{
  delete ocnode_buffer;
  delete facelist_buffer;
  delete bbox_buffer;
  delete range_buffer;

  delete placeholder;
  delete ocnode_map;
  delete face_map;
}


///////////////////////////////////////////////////////////////////////////////
/* virtual */ void
serialize_tree_dfs_traversal::visit ( ocnode& n ) const
{
  if ( n.has_children() ) // serialize inner node
  {
    serialize_inner_node ( n );
  } else {
    serialize_outer_node ( n );
  }
}


///////////////////////////////////////////////////////////////////////////////
void
serialize_tree_dfs_traversal::serialize_inner_node  ( ocnode& n ) const
{
  unsigned id = unsigned( ocnode_buffer->size() );
                
  unsigned range_id = serialize_range ( n.range().minimum(), n.range().maximum() );

  ocnode_buffer->push_back ( gpucast::gl::vec4u ( determine_nodetype(n), range_id, 0U, 0U ));
  ocnode_buffer->push_back ( gpucast::gl::vec4u ( unsigned(n.get_children()[0]->id()), unsigned(n.get_children()[1]->id()), unsigned(n.get_children()[2]->id()), unsigned(n.get_children()[3]->id()) ));
  ocnode_buffer->push_back ( gpucast::gl::vec4u ( unsigned(n.get_children()[4]->id()), unsigned(n.get_children()[5]->id()), unsigned(n.get_children()[6]->id()), unsigned(n.get_children()[7]->id()) ));
  
  placeholder->push_back ( id + 1 );
  placeholder->push_back ( id + 2 );

  ocnode_map->insert ( std::make_pair ( n.id(), id ) );

  std::for_each ( n.begin(), n.end(), std::bind ( &node::accept, std::placeholders::_1, std::cref(*this) ) );
}


///////////////////////////////////////////////////////////////////////////////
void
serialize_tree_dfs_traversal::serialize_outer_node  ( ocnode& n ) const
{
  unsigned ocnode_id = unsigned ( ocnode_buffer->size() );
  unsigned face_id   = unsigned ( facelist_buffer->size() );
  unsigned range_id = serialize_range ( n.range().minimum(), n.range().maximum() );

  gpucast::gl::vec4u entry ( determine_nodetype(n), range_id, face_id, unsigned(n.faces()) );

  std::for_each ( n.face_begin(), n.face_end(), std::bind ( &serialize_tree_dfs_traversal::serialize_face, this, std::placeholders::_1 ) );

  ocnode_buffer->push_back ( entry );
  ocnode_map->insert ( std::make_pair ( n.id(), ocnode_id ) );
}


///////////////////////////////////////////////////////////////////////////////
unsigned                          
serialize_tree_dfs_traversal::determine_nodetype ( ocnode& n ) const
{
#if 1
  unsigned type = 0;
  type += 2 * (!n.has_children());
  type += n.contains_outer_face();
#else
  unsigned type = n.has_children() ? 0 : 1;
#endif
  return type;
}


///////////////////////////////////////////////////////////////////////////////
void                        
serialize_tree_dfs_traversal::serialize_face ( face_ptr const& f ) const
{
  if ( face_map->count ( f ) ) // push known face entry
  {
    facelist_buffer->push_back ( face_map->at(f) );
  } else { // create new entry
    gpucast::gl::vec4u face_entry ( f->surface_id, 
                             serialize_range ( f->attribute_range.minimum(), f->attribute_range.maximum() ), 
                             serialize_bbox  ( f->obb ), 
                             f->outer );

    facelist_buffer->push_back ( face_entry );

    // store for later usage
    face_map->insert ( std::make_pair ( f, face_entry ) );
  }
}


///////////////////////////////////////////////////////////////////////////////
unsigned
  serialize_tree_dfs_traversal::serialize_bbox ( gpucast::math::obbox3f const& b ) const
{
  unsigned id = unsigned(bbox_buffer->size());
 
  auto m_inverse = b.orientation().inverse();

  bbox_buffer->push_back ( gpucast::gl::vec4f ( b.orientation().col(0)[0], b.orientation().col(0)[1], b.orientation().col(0)[2], 0.0f ) );
  bbox_buffer->push_back ( gpucast::gl::vec4f ( b.orientation().col(1)[0], b.orientation().col(1)[1], b.orientation().col(1)[2], 0.0f ) );
  bbox_buffer->push_back ( gpucast::gl::vec4f ( b.orientation().col(2)[0], b.orientation().col(2)[1], b.orientation().col(2)[2], 0.0f ) );
  bbox_buffer->push_back ( gpucast::gl::vec4f ( 0.0f, 0.0f, 0.0f, 1.0f ) );
  
  bbox_buffer->push_back ( gpucast::gl::vec4f ( m_inverse.col(0)[0], m_inverse.col(0)[1], m_inverse.col(0)[2], 0.0f ) );
  bbox_buffer->push_back ( gpucast::gl::vec4f ( m_inverse.col(1)[0], m_inverse.col(1)[1], m_inverse.col(1)[2], 0.0f ) );
  bbox_buffer->push_back ( gpucast::gl::vec4f ( m_inverse.col(2)[0], m_inverse.col(2)[1], m_inverse.col(2)[2], 0.0f ) );
  bbox_buffer->push_back ( gpucast::gl::vec4f ( 0.0f,                0.0f,                0.0f,                1.0f ) );

  bbox_buffer->push_back ( gpucast::gl::vec4f ( b.low() ) );
  bbox_buffer->push_back ( gpucast::gl::vec4f ( b.high() ) );
 
  bbox_buffer->push_back ( gpucast::gl::vec4f ( b.center() ) );

  return id;
}


///////////////////////////////////////////////////////////////////////////////
unsigned
serialize_tree_dfs_traversal::serialize_range ( float min, float max ) const
{
  unsigned id = unsigned(range_buffer->size());

  range_buffer->push_back (min);
  range_buffer->push_back (max);

  return id;
}


///////////////////////////////////////////////////////////////////////////////
void
serialize_tree_dfs_traversal::finalize () const
{
  if ( placeholder->empty() ) return;

  for ( auto i = placeholder->begin(); i != placeholder->end(); ++i )
  {
    // replace unique id with buffer id
    gpucast::gl::vec4u entry = (*ocnode_buffer)[*i];

    entry[0] = ocnode_map->at(entry[0]);
    entry[1] = ocnode_map->at(entry[1]);
    entry[2] = ocnode_map->at(entry[2]);
    entry[3] = ocnode_map->at(entry[3]);

    (*ocnode_buffer)[*i] = entry;
  }

  placeholder->clear();
}


///////////////////////////////////////////////////////////////////////////////
std::vector<gpucast::gl::vec4u> const& 
serialize_tree_dfs_traversal::nodebuffer () const
{
  finalize();
  return *ocnode_buffer;
}


///////////////////////////////////////////////////////////////////////////////
std::vector<gpucast::gl::vec4u> const& 
serialize_tree_dfs_traversal::facelistbuffer () const
{
  return *facelist_buffer;
}


///////////////////////////////////////////////////////////////////////////////
std::vector<gpucast::gl::vec4f> const& 
serialize_tree_dfs_traversal::bboxbuffer () const
{
  return *bbox_buffer;
}


///////////////////////////////////////////////////////////////////////////////
std::vector<float> const&       
serialize_tree_dfs_traversal::limitbuffer () const
{
  return *range_buffer;
}


} // namespace gpucast