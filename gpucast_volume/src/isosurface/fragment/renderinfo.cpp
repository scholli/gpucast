/********************************************************************************
*
* Copyright (C) 2007-2012 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : isosurface/fragment/renderinfo.cpp
*  project    : gpucast
*  description:
*
********************************************************************************/

// header i/f
#include "gpucast/volume/isosurface/fragment/renderinfo.hpp"

#include <gpucast/math/interval.hpp>

#include <gpucast/volume/isosurface/fragment/renderbin.hpp>
#include <gpucast/volume/isosurface/fragment/split_heuristic.hpp>


namespace gpucast {

////////////////////////////////////////////////////////////////////////////////
renderinfo::renderinfo()
: _range      ( -std::numeric_limits<value_type>::max(), std::numeric_limits<value_type>::max() ),
  _outerbin   (),
  _renderbins (),
  _indices    ()
{}


////////////////////////////////////////////////////////////////////////////////
renderinfo::renderinfo( interval_type const& range )
: _range      ( range ),
  _outerbin   (),
  _renderbins (),
  _indices    ()
{
  _renderbins.push_back ( renderbin ( range ) );
}


////////////////////////////////////////////////////////////////////////////////
renderinfo::~renderinfo()
{}


////////////////////////////////////////////////////////////////////////////////
std::size_t
renderinfo::size() const
{
  return _indices.size();
}


////////////////////////////////////////////////////////////////////////////////
void
renderinfo::clear() 
{
  return _renderbins.clear();
}


////////////////////////////////////////////////////////////////////////////////
std::vector<int>::const_iterator  
renderinfo::begin () const
{
  return _indices.begin();
}


////////////////////////////////////////////////////////////////////////////////
std::vector<int>::const_iterator  
renderinfo::end () const
{
  return _indices.end();
}


////////////////////////////////////////////////////////////////////////////////
renderinfo::interval_type const&              
renderinfo::range ( ) const
{
  return _range;
}


////////////////////////////////////////////////////////////////////////////////
void                              
renderinfo::range( interval_type const& r )
{
  _range = r;
}


////////////////////////////////////////////////////////////////////////////////
void                      
renderinfo::insert ( renderchunk_ptr const& chunk )
{
  // if chunk is part of outer surface, put it into special bin
  if ( chunk->outer ) 
  {
    _outerbin.insert ( chunk );
  } else { // else sort chunk 
    for ( auto i = _renderbins.begin(); i != _renderbins.end(); ++i )
    {
      if ( i->range().overlap(chunk->range) ) 
      {
        i->insert ( chunk );
      } 
    }
  }
}


////////////////////////////////////////////////////////////////////////////////
void                              
renderinfo::optimize ( split_heuristic const& h )
{
  for ( renderbin_container::iterator bin = _renderbins.begin(); bin != _renderbins.end(); )
  {
    // check if renderbin can be split
    float split_pos = 0.0f;
    if ( h.splitable ( *bin, split_pos ) )
    {
      // split renderbin
      renderbin lhs, rhs;
      h.split( *bin, lhs, rhs, split_pos );

      // erase bin that was split
      _renderbins.erase ( bin );

      // insert split result
      _renderbins.push_back ( lhs );
      _renderbins.push_back ( rhs );

      // restart split algorithm
      bin = _renderbins.begin();
    } else {
      ++bin;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
void                      
renderinfo::serialize ()
{
  _indices.clear();

  _outerbin.baseindex ( int(_indices.size()) );
  _outerbin.indices   ( 0 );

  for ( renderbin::renderchunk_const_iterator chunk = _outerbin.begin(); chunk != _outerbin.end(); ++chunk )
  {
    std::size_t offset = _indices.size();
    _indices.resize ( _indices.size() + (*chunk)->indices.size() );
    std::copy ( (*chunk)->indices.begin(), (*chunk)->indices.end(), _indices.begin() + offset );

    _outerbin.indices ( _outerbin.indices() + int((*chunk)->indices.size()) );
  }

  for ( renderbin& bin : _renderbins )
  {
    bin.baseindex ( int (_indices.size() ) );
    bin.indices ( 0 );

    for ( renderbin::renderchunk_const_iterator chunk = bin.begin(); chunk != bin.end(); ++chunk )
    {
      std::size_t offset = _indices.size();
      _indices.resize ( _indices.size() + (*chunk)->indices.size() );
      std::copy ( (*chunk)->indices.begin(), (*chunk)->indices.end(), _indices.begin() + offset );

      bin.indices ( bin.indices() + int((*chunk)->indices.size()) );
    }
  }
}


/////////////////////////////////////////////////////////////////////////////////
std::size_t
renderinfo::renderbins_size () const
{
  return _renderbins.size();
}


/////////////////////////////////////////////////////////////////////////////////
renderinfo::renderbin_const_iterator          
renderinfo::renderbin_begin ( ) const
{
  return _renderbins.begin();
}


/////////////////////////////////////////////////////////////////////////////////
renderinfo::renderbin_const_iterator          
renderinfo::renderbin_end ( ) const
{
  return _renderbins.end();
}


////////////////////////////////////////////////////////////////////////////////
bool                      
renderinfo::get_renderbin ( value_type isovalue, int& index, int& size ) const
{
  for ( renderbin const& bin : _renderbins )
  {
    if ( bin.range().in ( isovalue ) )
    {
      index = bin.baseindex();
      size  = bin.indices();
      return true;
    }
  }
  return false;
}


////////////////////////////////////////////////////////////////////////////////
void                      
renderinfo::get_outerbin ( int& index, int& size ) const
{
  index = _outerbin.baseindex();
  size  = _outerbin.indices();
}


////////////////////////////////////////////////////////////////////////////////
void                              
renderinfo::write ( std::ostream& os ) const
{
  // save minimum and maximum for total range
  value_type amin = _range.minimum();
  value_type amax = _range.maximum();

  os.write ( reinterpret_cast<char const*> ( &amin ), sizeof ( value_type ) );
  os.write ( reinterpret_cast<char const*> ( &amax ), sizeof ( value_type ) );

  // gather all chunks in a set
  std::set<renderchunk const*> chunks;
  for ( renderbin::renderchunk_const_iterator i = _outerbin.begin(); i != _outerbin.end(); ++i ) 
  {
    chunks.insert(i->get());
  }

  for ( renderbin_container::const_iterator c = _renderbins.begin(); c != _renderbins.end(); ++c ) 
  {
    for ( renderbin::renderchunk_const_iterator i = c->begin(); i != c->end(); ++i ) 
    {
      chunks.insert(i->get());
    }
  }

  // write chunks with unique id
  std::size_t nchunks = chunks.size();
  os.write ( reinterpret_cast<char const*> (&nchunks), sizeof ( std::size_t ) );

  for ( std::set<renderchunk const*>::const_iterator i = chunks.begin(); i != chunks.end(); ++i )
  {
    std::size_t uid = std::size_t(*i);
    os.write ( reinterpret_cast<char const*> (&uid), sizeof ( std::size_t ) );
    (*i)->write(os);
  }

  // write outerbin
  int         baseindex      = _outerbin.baseindex();
  int         indices        = _outerbin.indices();
  value_type  bin_min        = _outerbin.range().minimum();
  value_type  bin_max        = _outerbin.range().maximum();
  std::size_t nchunks_in_bin = _outerbin.chunks();

  os.write ( reinterpret_cast<char const*> (&baseindex),      sizeof ( int ) );
  os.write ( reinterpret_cast<char const*> (&indices),        sizeof ( int ) );
  os.write ( reinterpret_cast<char const*> (&bin_min),        sizeof ( value_type ) );
  os.write ( reinterpret_cast<char const*> (&bin_max),        sizeof ( value_type ) );
  os.write ( reinterpret_cast<char const*> (&nchunks_in_bin), sizeof ( std::size_t ) );

  for ( renderbin::renderchunk_const_iterator i = _outerbin.begin(); i != _outerbin.end(); ++i ) 
  {
    std::size_t uid = std::size_t(i->get());
    os.write ( reinterpret_cast<char const*> (&uid), sizeof ( std::size_t ) );
  }

  std::size_t nbins = _renderbins.size();
  os.write ( reinterpret_cast<char const*> (&nbins), sizeof ( std::size_t ) );

  for ( renderbin_container::const_iterator c = _renderbins.begin(); c != _renderbins.end(); ++c ) 
  {
    baseindex      = c->baseindex();
    indices        = c->indices();
    bin_min        = c->range().minimum();
    bin_max        = c->range().maximum();
    nchunks_in_bin = c->chunks();

    os.write ( reinterpret_cast<char const*> (&baseindex),      sizeof ( int ) );
    os.write ( reinterpret_cast<char const*> (&indices),        sizeof ( int ) );
    os.write ( reinterpret_cast<char const*> (&bin_min),        sizeof ( value_type ) );
    os.write ( reinterpret_cast<char const*> (&bin_max),        sizeof ( value_type ) );
    os.write ( reinterpret_cast<char const*> (&nchunks_in_bin), sizeof ( std::size_t ) );

    for ( renderbin::renderchunk_const_iterator i = c->begin(); i != c->end(); ++i ) 
    {
      std::size_t uid = std::size_t(i->get());
      os.write ( reinterpret_cast<char const*> (&uid), sizeof ( std::size_t ) );
    }
  }
  
  std::size_t nindices = _indices.size();
  os.write ( reinterpret_cast<char const*> (&nindices), sizeof ( std::size_t ) );
  os.write ( reinterpret_cast<char const*> (&_indices.front()), nindices * sizeof (int) );
}


////////////////////////////////////////////////////////////////////////////////
void                              
renderinfo::read ( std::istream& is )
{
  clear();

  // save minimum and maximum for total range
  value_type amin;
  value_type amax;

  is.read ( reinterpret_cast<char*> ( &amin ), sizeof ( value_type ) );
  is.read ( reinterpret_cast<char*> ( &amax ), sizeof ( value_type ) );

  _range = interval_type ( amin, amax, gpucast::math::included, gpucast::math::included );

  // gather all chunks in a set
  std::map<std::size_t, renderchunk_ptr> chunks;

  // write chunks with unique id
  std::size_t nchunks;
  is.read ( reinterpret_cast<char*> (&nchunks), sizeof ( std::size_t ) );

  for ( unsigned i = 0; i != nchunks; ++i )
  {
    std::size_t uid;
    is.read ( reinterpret_cast<char*> (&uid), sizeof ( std::size_t ) );
    chunks.insert(std::make_pair(uid, renderchunk_ptr(new renderchunk)));
    chunks[uid]->read(is);
  }

  // write outerbin
  int         baseindex;     
  int         indices;       
  value_type  bin_min;      
  value_type  bin_max;       
  std::size_t nchunks_in_bin;

  is.read ( reinterpret_cast<char*> (&baseindex),      sizeof ( int ) );
  is.read ( reinterpret_cast<char*> (&indices),        sizeof ( int ) );
  is.read ( reinterpret_cast<char*> (&bin_min),        sizeof ( value_type ) );
  is.read ( reinterpret_cast<char*> (&bin_max),        sizeof ( value_type ) );
  is.read ( reinterpret_cast<char*> (&nchunks_in_bin), sizeof ( std::size_t ) );

  _outerbin.range     ( interval_type ( bin_min, bin_max, gpucast::math::included, gpucast::math::included ) );
  _outerbin.baseindex ( baseindex );
  _outerbin.indices   ( indices );

  for ( unsigned i = 0; i != nchunks_in_bin; ++i )
  {
    std::size_t uid;
    is.read ( reinterpret_cast<char*> (&uid), sizeof ( std::size_t ) );
    _outerbin.insert ( chunks[uid] );
  }

  std::size_t nbins;
  is.read ( reinterpret_cast<char*> (&nbins), sizeof ( std::size_t ) );

  for ( unsigned i = 0; i != nbins; ++i )
  {
    is.read ( reinterpret_cast<char*> (&baseindex),      sizeof ( int ) );
    is.read ( reinterpret_cast<char*> (&indices),        sizeof ( int ) );
    is.read ( reinterpret_cast<char*> (&bin_min),        sizeof ( value_type ) );
    is.read ( reinterpret_cast<char*> (&bin_max),        sizeof ( value_type ) );
    is.read ( reinterpret_cast<char*> (&nchunks_in_bin), sizeof ( std::size_t ) );

    renderbin bin ( interval_type ( bin_min, bin_max ) );
    bin.baseindex ( baseindex );
    bin.indices ( indices );

    for ( unsigned j = 0; j != nchunks_in_bin; ++j )
    {
      std::size_t uid;
      is.read ( reinterpret_cast<char*> (&uid), sizeof ( std::size_t ) );
      bin.insert ( chunks[uid] );
    }
    _renderbins.push_back ( bin );
  }
  
  std::size_t nindices;
  is.read ( reinterpret_cast<char*> (&nindices), sizeof ( std::size_t ) );
  _indices.resize ( nindices );

  is.read ( reinterpret_cast<char*> (&_indices.front()), nindices * sizeof (int) );
}

} // namespace gpucast

