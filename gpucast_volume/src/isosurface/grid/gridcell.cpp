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
#include "gpucast/volume/isosurface/grid/gridcell.hpp"

// header, system

// header, external

// header, project

namespace gpucast {

  /////////////////////////////////////////////////////////////////////////////
  gridcell::gridcell()
    : _range (),
      _faces ()  
  {}

  /////////////////////////////////////////////////////////////////////////////
  void 
  gridcell::clear()
  {
    _range = attribute_interval_t();
    _faces.clear();
  }

  /////////////////////////////////////////////////////////////////////////////
  void 
  gridcell::add ( face_ptr const& face )
  {
    // set or merge attribute bounds of cell
    if ( _faces.empty() )
    {
      _range = face->attribute_range; 
    } else {
      _range.merge(face->attribute_range); 
    }

    // store face in gridcell 
    _faces.push_back( face );
  }
  
  /////////////////////////////////////////////////////////////////////////////
  float gridcell::attribute_min () const
  {
    return _range.minimum();
  }

  /////////////////////////////////////////////////////////////////////////////
  float gridcell::attribute_max () const
  {
    return _range.maximum();
  }

  /////////////////////////////////////////////////////////////////////////////
  std::vector<face_ptr>::const_iterator gridcell::begin() const
  {
    return _faces.begin();
  }

  /////////////////////////////////////////////////////////////////////////////
  std::vector<face_ptr>::const_iterator gridcell::end() const
  {
    return _faces.end();
  }

  /////////////////////////////////////////////////////////////////////////////
  std::size_t gridcell::faces () const
  {
    return _faces.size();
  }

} // namespace gpucast

