/********************************************************************************
*
* Copyright (C) 2007-2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : nurbsvolume.cpp
*  project    : gpucast
*  description:
*
********************************************************************************/

// header i/f
#include "gpucast/volume/nurbsvolume.hpp"



namespace gpucast {

////////////////////////////////////////////////////////////////////////////////
nurbsvolume::nurbsvolume()
  : base_type         (),
    _data             (),
    _boundary_outer   (),
    _boundary_special ()
{
  std::fill( _boundary_outer.begin(),    _boundary_outer.end(),    true );
  std::fill( _boundary_special.begin(),  _boundary_special.end(),  false );
}


////////////////////////////////////////////////////////////////////////////////
nurbsvolume::~nurbsvolume()
{}


////////////////////////////////////////////////////////////////////////////////
void                  
nurbsvolume::clear ()
{
  base_type::clear();
  _data.clear();
}


////////////////////////////////////////////////////////////////////////////////
void 
nurbsvolume::attach( std::string const& name, attribute_volume_type const& data, bool normalize, value_type const& s )
{
  _data.insert ( std::make_pair ( name, data ) );

  if ( normalize ) {
    _data[name].normalize();
  } else {
    if ( s != 1) {
      _data[name].scale(s);
    }
  }
}


////////////////////////////////////////////////////////////////////////////////
nurbsvolume::const_map_iterator    
nurbsvolume::data ( std::string const& name ) const
{
  return _data.find(name);
}


////////////////////////////////////////////////////////////////////////////////
nurbsvolume::const_map_iterator    
nurbsvolume::data_begin () const
{
  return _data.begin();
}


////////////////////////////////////////////////////////////////////////////////
nurbsvolume::const_map_iterator    
nurbsvolume::data_end () const
{
  return _data.end();
}


////////////////////////////////////////////////////////////////////////////////
void                                
nurbsvolume::crop ( std::string const& attribute_name )
{
  for ( std::map<std::string, attribute_volume_type>::iterator i = _data.begin(); i != _data.end(); ) 
  {
    if ( i->first != attribute_name ) 
    {
      std::cout << i->first << " cropped." << std::endl;
      _data.erase(i++);
    } else {
      std::cout << i->first << " kept." << std::endl;
      ++i;
    }
  }
}


////////////////////////////////////////////////////////////////////////////////
void                                
nurbsvolume::normalize_attribute ()
{
  std::for_each(_data.begin(), _data.end(), [] ( attribute_map_type::value_type& v ) { v.second.normalize(); } ); 
}


////////////////////////////////////////////////////////////////////////////////
void                  
nurbsvolume::write ( std::ostream& os ) const
{
  base_type::write(os);

  std::size_t attributes = _data.size();
  os.write ( reinterpret_cast<char const*> (&attributes), sizeof ( std::size_t ) );

  for ( attribute_map_type::value_type const& v : _data ) 
  {
    std::size_t attribute_name_length = v.first.size();
    os.write ( reinterpret_cast<char const*> (&attribute_name_length), sizeof ( std::size_t) );
    os.write ( v.first.c_str(), attribute_name_length );
    v.second.write(os);
  }

  os.write (reinterpret_cast<char const*> (&_boundary_outer[0]),    sizeof(_boundary_outer));
  os.write (reinterpret_cast<char const*> (&_boundary_special[0]),  sizeof(_boundary_special));
}


////////////////////////////////////////////////////////////////////////////////
void                  
nurbsvolume::read ( std::istream& is )
{
  base_type::clear();
  _data.clear();

  base_type::read(is);

  std::size_t attributes;
  is.read ( reinterpret_cast<char*> (&attributes), sizeof ( std::size_t ) );

  for ( unsigned i = 0; i != attributes; ++i )
  {
    std::size_t attribute_name_length;
    is.read ( reinterpret_cast<char*> (&attribute_name_length), sizeof ( std::size_t) );

    std::string attribute_name;
    attribute_name.resize(attribute_name_length);
    is.read ( reinterpret_cast<char*> (&attribute_name[0]), attribute_name_length );

    _data.insert ( std::make_pair ( attribute_name, attribute_volume_type() ) );
    _data[attribute_name].read(is);
  }

  is.read (reinterpret_cast<char*> (&_boundary_outer[0]),   sizeof(_boundary_outer));
  is.read (reinterpret_cast<char*> (&_boundary_special[0]), sizeof(_boundary_special));
}


////////////////////////////////////////////////////////////////////////////////
std::array<bool, 6> const&        
nurbsvolume::is_outer () const
{
  return _boundary_outer;
}


////////////////////////////////////////////////////////////////////////////////
void                                
nurbsvolume::is_outer ( std::array<bool, 6> const& b )
{
  _boundary_outer = b;
}


////////////////////////////////////////////////////////////////////////////////
std::array<bool, 6> const&        
nurbsvolume::is_special () const
{
  return _boundary_special;
}


////////////////////////////////////////////////////////////////////////////////
void                                
nurbsvolume::is_special ( std::array<bool, 6> const& b )
{
  _boundary_special = b;
}



} // namespace gpucast
