/*******************************************************************************
*
* Copyright (C) 2007-2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : igs_loader.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_IGS_LOADER_HPP
#define GPUCAST_IGS_LOADER_HPP

// header, system
#include <string>
#include <vector>
#include <memory>



// header, project
#include <gpucast/core/gpucast.hpp>

namespace gpucast {

// fwd decl
class nurbssurfaceobject;

///////////////////////////////////////////////////////////////////////////////
class GPUCAST_CORE igs_loader
{
public : // typedefs 

  typedef std::shared_ptr<nurbssurfaceobject> nurbsobject_ptr;

public: // non-copyable

  igs_loader() = default;
  ~igs_loader() = default;

  igs_loader (igs_loader const& cpy) = delete;
  igs_loader& operator= (igs_loader const& cpy) = delete;

public : // methods

  nurbsobject_ptr     load (std::string const& file);
  std::string         error_message () const;

private: // methods

  bool                _load(std::fstream& is);
  
private : // members

  std::string                         _error;
  std::shared_ptr<nurbssurfaceobject> _result;
};

} // namespace gpucast

#endif // GPUCAST_IGS_LOADER_HPP
