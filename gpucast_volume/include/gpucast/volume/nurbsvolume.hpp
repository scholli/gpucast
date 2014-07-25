  /********************************************************************************
*
* Copyright (C) 2007-2010 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : nurbsvolume.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_NURBSVOLUME_HPP
#define GPUCAST_NURBSVOLUME_HPP

#ifdef _MSC_VER
 #pragma warning(disable: 4251)
#endif

// header, system
#include <map>
#include <string>

// header, external
#include <gpucast/math/parametric/nurbsvolume.hpp>

// header, project
#include <gpucast/volume/gpucast.hpp>

namespace gpucast 
{

class GPUCAST_VOLUME nurbsvolume : public gpucast::math::nurbsvolume<gpucast::math::point<float,3> >
{
public : // enums, typedefs

  enum boundary_t { umin  = 0, 
                    umax  = 1,                   
                    vmin  = 2, 
                    vmax  = 3, 
                    wmin  = 4, 
                    wmax  = 5,
                    count = 6};

  typedef gpucast::math::nurbsvolume<point_type>                  base_type;
  typedef gpucast::math::point<float, 1>                          attribute_type;
  typedef gpucast::math::nurbsvolume<attribute_type>              attribute_volume_type;
  typedef std::map<std::string, attribute_volume_type>  attribute_map_type;
  typedef attribute_map_type::const_iterator            const_map_iterator;
  typedef std::array<bool, 6>                         bool6_t;

public : // c'tor / d'tor

  nurbsvolume();
  ~nurbsvolume();

public : // methods
                                                 
  void                                clear           ();
                                                      
  void                                attach          ( std::string const&            name, 
                                                        attribute_volume_type const&  data,
                                                        bool                          normalize = false,
                                                        value_type const&             scale = 1 );
  
  const_map_iterator                  data            ( std::string const& name ) const;
                                                      
  const_map_iterator                  data_begin      () const;
  const_map_iterator                  data_end        () const;
                                                      
  void                                crop            ( std::string const& );
  void                                normalize_attribute ();

  void                                write           ( std::ostream& os ) const;
  void                                read            ( std::istream& is );
                                                      
  bool6_t const&                      is_outer        () const;
  void                                is_outer        ( bool6_t const& );

  bool6_t const&                      is_special      () const;
  void                                is_special      ( bool6_t const& );

private : // data members

  attribute_map_type      _data;

  std::array<bool, 6>   _boundary_outer;
  std::array<bool, 6>   _boundary_special;
};

} // namespace gpucast

#endif // GPUCAST_NURBSVOLUME_HPP
