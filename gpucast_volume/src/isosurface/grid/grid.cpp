/********************************************************************************
*
* Copyright (C) 2007-2013 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : gridcell.cpp
*  project    : gpucast
*  description:
*
********************************************************************************/

// header i/f
#include "gpucast/volume/isosurface/grid/grid.hpp"

#include <gpucast/core/util.hpp>

namespace gpucast {

  /////////////////////////////////////////////////////////////////////////////
  grid::grid ( unsigned width, unsigned height, unsigned depth )
    : _object       (),
      _boundingbox  (),
      _size         (),
      _cells        (),
      _unique_faces (0)
  {
    _size[0] = width;
    _size[1] = height;
    _size[2] = depth;

    _cells.resize(width * height * depth);
  }
  
  /////////////////////////////////////////////////////////////////////////////
  void                      
  grid::generate ( volume_ptr const&            volume,
                   std::vector<face_ptr> const& serialized_faces )
  {
    _unique_faces         = unsigned(serialized_faces.size());
    _object               = volume;
    _boundingbox          = _object->bbox();
    gpucast::math::point3f cellsize = _boundingbox.size() / gpucast::math::point3f(float(_size[0]), float(_size[1]), float(_size[2]));

    std::cout << "Creating grid : " << _size[0] << " x " << _size[1] << " x " << _size[2] << " for " << serialized_faces.size() << " faces.\n";
    unsigned facecount    = 0;
    
    for ( auto face = serialized_faces.begin(); face != serialized_faces.end(); ++face, ++facecount )
    {
      gpucast::math::bbox3f  aabb     = (**face).obb.aabb();
      
      // caution: OBB might overlap total AABB -> TODO : implement optimized version
#if 0
      std::size_t start_u   = 0;// std::max ( int ( floor((aabb.min[0] - _boundingbox.min[0]) / cellsize[0])), 0 ); 
      std::size_t start_v   = 0;// std::max ( int ( floor((aabb.min[1] - _boundingbox.min[1]) / cellsize[1])), 0 ); 
      std::size_t start_w   = 0;// std::max ( int ( floor((aabb.min[2] - _boundingbox.min[2]) / cellsize[2])), 0 );                       
                               
      std::size_t end_u     = _size[0];// std::min ( int ( ceil((aabb.max[0] - _boundingbox.min[0]) / cellsize[0])), int(_size[0]-1) ); 
      std::size_t end_v     = _size[1];// std::min ( int ( ceil((aabb.max[1] - _boundingbox.min[1]) / cellsize[1])), int(_size[1]-1) ); 
      std::size_t end_w     = _size[2];// std::min ( int ( ceil((aabb.max[2] - _boundingbox.min[2]) / cellsize[2])), int(_size[2]-1) ); 
#else
      std::size_t start_u   = std::max ( int ( floor((aabb.min[0] - _boundingbox.min[0]) / cellsize[0])), 0 ); 
      std::size_t start_v   = std::max ( int ( floor((aabb.min[1] - _boundingbox.min[1]) / cellsize[1])), 0 ); 
      std::size_t start_w   = std::max ( int ( floor((aabb.min[2] - _boundingbox.min[2]) / cellsize[2])), 0 );                       
                               
      std::size_t end_u     = std::min ( int ( ceil((aabb.max[0] - _boundingbox.min[0]) / cellsize[0])), int(_size[0]) ); 
      std::size_t end_v     = std::min ( int ( ceil((aabb.max[1] - _boundingbox.min[1]) / cellsize[1])), int(_size[1]) ); 
      std::size_t end_w     = std::min ( int ( ceil((aabb.max[2] - _boundingbox.min[2]) / cellsize[2])), int(_size[2]) ); 

      if ( start_u > end_u || 
           start_v > end_v ||
           start_w > end_w)
      {
        std::size_t start_u   = 0;
        std::size_t start_v   = 0;
        std::size_t start_w   = 0;

        std::size_t end_u     = _size[0];
        std::size_t end_v     = _size[1];
        std::size_t end_w     = _size[2];
      }
#endif
      std::cout << "Assigning faces to cells : " << 100.0f * float(facecount)/serialized_faces.size() << " %    \r";
      for ( std::size_t u = start_u; u < end_u; ++u ) 
      {
        for ( std::size_t v = start_v; v < end_v; ++v ) 
        {
          for ( std::size_t w = start_w; w < end_w; ++w ) 
          {
            gpucast::math::bbox3f cellbox (_boundingbox.min + cellsize * gpucast::math::point3f(float(u), float(v), float(w)),
                                 _boundingbox.min + cellsize * gpucast::math::point3f(float(u+1), float(v+1), float(w+1)));

            if ( (**face).obb.overlap ( cellbox ) )
            {
              (*this)(u, v, w).add (*face);
            }
          }
        }
      }
    }

    print_info ( std::cout );

  }


  /////////////////////////////////////////////////////////////////////////////
  void                                
  grid::serialize ( std::vector<gpucast::gl::vec4u>& cellbuffer,
                    std::vector<gpucast::gl::vec4u>& facebuffer,
                    std::vector<gpucast::gl::vec4f>& bboxbuffer ) const
  {
    cellbuffer.clear();
    facebuffer.clear();
    bboxbuffer.clear();

    std::map<face_ptr, unsigned> face_map;

    for ( auto cell_iter = _cells.begin(); cell_iter != _cells.end(); ++cell_iter )
    {
      serialize ( *cell_iter, cellbuffer, facebuffer, bboxbuffer, face_map );
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  unsigned                            
  grid::serialize ( gridcell const&               cell, 
                    std::vector<gpucast::gl::vec4u>&     cellbuffer,
                    std::vector<gpucast::gl::vec4u>&     facebuffer,
                    std::vector<gpucast::gl::vec4f>&     bboxbuffer,
                    std::map<face_ptr, unsigned>& face_map ) const
  {
    unsigned first_face_id = facebuffer.size();

    // count if there are outer faces in cell
    unsigned outer_faces   = 0;
    std::for_each ( cell.begin(), cell.end(), [&outer_faces] ( face_ptr const& f ) { outer_faces += int(f->outer); } );
    
    gpucast::gl::vec4u cell_entry (bit_cast<float,unsigned>(cell.attribute_min()),
                            bit_cast<float,unsigned>(cell.attribute_max()),
                            uint2ToUInt ( cell.faces(), outer_faces ),
                            first_face_id );

    cellbuffer.push_back ( cell_entry );

    if ( cell.faces() != 0 ) 
    {
      for ( auto face_iter = cell.begin(); face_iter != cell.end(); ++face_iter )
      {
        unsigned face_data_id  = serialize ( *face_iter, bboxbuffer, face_map );
        gpucast::gl::vec4u face_entry ( face_data_id, 
                                 (**face_iter).outer,
                                 (**face_iter).surface_id,
                                 0 );

        facebuffer.push_back (face_entry);
      }
    } else {
      facebuffer.push_back ( gpucast::gl::vec4u (0,0,0,0) );
    }

    return first_face_id;
  }

  /////////////////////////////////////////////////////////////////////////////
  unsigned                            
  grid::serialize ( face_ptr const& face, 
                    std::vector<gpucast::gl::vec4f>& bboxbuffer,
                    std::map<face_ptr, unsigned>& face_map ) const
  {
    auto face_iter = face_map.find(face);
    if ( face_iter != face_map.end() ) {
      return face_iter->second;
    } else {
      unsigned face_id = serialize ( face->obb, bboxbuffer );
      face_map.insert(std::make_pair(face, face_id));
      return face_id;
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  unsigned                            
  grid::serialize ( gpucast::math::obbox3f const& b,
                    std::vector<gpucast::gl::vec4f>& bboxbuffer ) const
  {
    unsigned id = unsigned(bboxbuffer.size());
 
    auto m_inverse = b.orientation().inverse();

    bboxbuffer.push_back ( gpucast::gl::vec4f ( b.orientation().col(0)[0], b.orientation().col(0)[1], b.orientation().col(0)[2], 0.0f ) );
    bboxbuffer.push_back ( gpucast::gl::vec4f ( b.orientation().col(1)[0], b.orientation().col(1)[1], b.orientation().col(1)[2], 0.0f ) );
    bboxbuffer.push_back ( gpucast::gl::vec4f ( b.orientation().col(2)[0], b.orientation().col(2)[1], b.orientation().col(2)[2], 0.0f ) );
    bboxbuffer.push_back ( gpucast::gl::vec4f ( 0.0f, 0.0f, 0.0f, 1.0f ) );
              
    bboxbuffer.push_back ( gpucast::gl::vec4f ( m_inverse.col(0)[0], m_inverse.col(0)[1], m_inverse.col(0)[2], 0.0f ) );
    bboxbuffer.push_back ( gpucast::gl::vec4f ( m_inverse.col(1)[0], m_inverse.col(1)[1], m_inverse.col(1)[2], 0.0f ) );
    bboxbuffer.push_back ( gpucast::gl::vec4f ( m_inverse.col(2)[0], m_inverse.col(2)[1], m_inverse.col(2)[2], 0.0f ) );
    bboxbuffer.push_back ( gpucast::gl::vec4f ( 0.0f,                0.0f,                0.0f,                1.0f ) );
              
    bboxbuffer.push_back ( gpucast::gl::vec4f ( b.low() ) );
    bboxbuffer.push_back ( gpucast::gl::vec4f ( b.high() ) );
              
    bboxbuffer.push_back ( gpucast::gl::vec4f ( b.center() ) );

    return id;
  }


  /////////////////////////////////////////////////////////////////////////////
  std::array<unsigned, 3> const&   
  grid::size () const
  {
    return _size;
  }

  /////////////////////////////////////////////////////////////////////////////
  gridcell const&                    
  grid::operator() (std::size_t u, std::size_t v, std::size_t w) const
  {
    assert ( u < _size[0] && v < _size[1] && w < _size[2] );
    return _cells[w * (_size[0] * _size[1]) + v * _size[0] + u];
  }

  /////////////////////////////////////////////////////////////////////////////
  gridcell&                    
  grid::operator() (std::size_t u, std::size_t v, std::size_t w)
  {
    assert ( u < _size[0] && v < _size[1] && w < _size[2] );
    return _cells[w * (_size[0] * _size[1]) + v * _size[0] + u];
  }

  /////////////////////////////////////////////////////////////////////////////
  void                                
  grid::print_info ( std::ostream& os ) const
  {
    unsigned max_faces_per_cell    = 0;
    
    unsigned empty_cells           = 0;
    unsigned total_face_references = 0;
    unsigned total_cells = _size[0] * _size[1] * _size[2];

    for ( auto cell_iter = _cells.begin(); cell_iter != _cells.end(); ++cell_iter )
    {
      auto cell = *cell_iter;

      total_face_references += cell.faces();
      if ( cell.faces() == 0 ) {
        ++empty_cells;
      } else {
        if ( cell.faces() > max_faces_per_cell ) 
        {
          max_faces_per_cell = cell.faces();
        }
        total_face_references += cell.faces();
      }
    }

    os << "resolution : " << _size[0] << "x" << _size[1] << "x" << _size[2] <<std::endl;
    os << " - #cells : " << total_cells << std::endl;
    os << " - #empty cells : " << empty_cells << std::endl;
    os << " - #faces : " << _unique_faces << std::endl;
    os << " - max faces per cell : " << max_faces_per_cell << std::endl;
    os << " - average faces per cell : " << float(total_face_references) / total_cells << std::endl;

  }



} // namespace gpucast
