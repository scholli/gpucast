/********************************************************************************
*
* Copyright (C) 2007-2013 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : grid.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_GRID_HPP
#define GPUCAST_GRID_HPP

// header, system
#include <array>
#include <vector>

// header, project
#include <gpucast/volume/gpucast.hpp>

#include <gpucast/volume/beziervolumeobject.hpp>
#include <gpucast/volume/isosurface/grid/gridcell.hpp>


namespace gpucast {

class GPUCAST_VOLUME grid 
{
  public : // typedefs / enums

    typedef beziervolumeobject                      volume_type;
    typedef std::shared_ptr<volume_type>          volume_ptr;

  public : // c'tor / d'tor

    grid ( unsigned width, unsigned height, unsigned depth );

  public : // methods

     // fill tree and build it
    void                                generate ( volume_ptr const&              volume,
                                                   std::vector<face_ptr> const&   serialized_faces );

    void                                serialize ( std::vector<gpucast::math::vec4u>&     cellbuffer,
                                                    std::vector<gpucast::math::vec4u>&     facebuffer,
                                                    std::vector<gpucast::math::vec4f>&     bboxbuffer ) const;

    unsigned                            serialize ( gridcell const&               cell, 
                                                    std::vector<gpucast::math::vec4u>&     cellbuffer,
                                                    std::vector<gpucast::math::vec4u>&     facebuffer,
                                                    std::vector<gpucast::math::vec4f>&     bboxbuffer,
                                                    std::map<face_ptr, unsigned>& face_map ) const;

    unsigned                            serialize ( face_ptr const&               face, 
                                                    std::vector<gpucast::math::vec4f>&     bboxbuffer,
                                                    std::map<face_ptr, unsigned>& face_map ) const;

    unsigned                            serialize ( gpucast::math::obbox3f const&           bbox,
                                                    std::vector<gpucast::math::vec4f>&     bboxbuffer ) const;

    std::array<unsigned, 3> const&      size () const;

    gridcell const&                     operator() (std::size_t u, std::size_t v, std::size_t w) const;
    gridcell&                           operator() (std::size_t u, std::size_t v, std::size_t w);

    void                                print_info ( std::ostream& os ) const;

  private : // auxilliary methods

  private : // attributes
    
    volume_ptr                      _object;
    volume_type::boundingbox_type   _boundingbox;

    unsigned                        _unique_faces;

    std::array<unsigned, 3>         _size;
    std::vector<gridcell>           _cells;
    
};

} // namespace gpucast

#endif // GPUCAST_GRID_HPP