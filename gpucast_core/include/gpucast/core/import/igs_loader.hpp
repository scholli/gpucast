/*******************************************************************************
*
* Copyright (C) 2007-2016 Bauhaus-Universitaet Weimar
*
*********************************************************************************
*
*  module     : igs_loader.hpp
*  project    : gpucast
*  description:
*
********************************************************************************/
#ifndef GPUCAST_IGS2_LOADER_HPP
#define GPUCAST_IGS2_LOADER_HPP

// header, system
#include <string>
#include <vector>
#include <memory>
#include <map>

// header, project
#include <gpucast/core/gpucast.hpp>


#include <gpucast/math/matrix4x4.hpp>
#include <gpucast/math/parametric/nurbscurve.hpp>
#include <gpucast/math/parametric/nurbssurface.hpp>
#include <gpucast/core/nurbssurface.hpp>

#include <boost/algorithm/string.hpp>

namespace gpucast {

  namespace iges {

    // IGES uses x.xxD-002 format instead of x.xxE-002
    GPUCAST_CORE double stod(std::string s);
    GPUCAST_CORE int stoi(std::string const& s); 

    const static unsigned LINE_LENGTH = 80;
    const static unsigned SECTION_INDEX_DIGITS = 7;
    const static unsigned DIRECTORY_COLUMN_WIDTH = 8;
    const static unsigned DIRECTORY_COLUMN_COUNT = 9;
    const static double MAX_COLOR_INTENSITY = 100.0;

    enum section_type {

      section_flag,
      section_start,
      section_global,
      section_directory,
      section_parameter,
      section_terminate
    };

    struct line {
      section_type type;  // section type
      std::size_t index;  // entity index
      std::size_t entity; // entity type
      std::string data;   // data
    };
    
    struct directory_entry {

      int entity_type;            // 1:01-08
      int parameter_data;         // 1:09-16
      int structure;              // 1:17-24
      int line_font_pattern;      // 1:25-32
      int level;                  // 1:33-40
      int view;                   // 1:41-48
      int transform;              // 1:49-56
      int label;                  // 1:57-64
      int status;                 // 1:65-72
      unsigned section_index;     // 1:73-80
                                  
      //int entity_type;          // 2:01-08
      int line_weight;            // 2:09-16
      int color_number;           // 2:17-24
      int parameter_line_count;   // 2:25-32
      int form_number;            // 2:33-40
      // unsigned reserved;       // 2:41-48
      // unsigned reserved;       // 2:49-56
      std::string entity_label;   // 2:57-64
      int entity_subscript;       // 2:65-72
      //unsigned sequence_number; // 2:73-80

      void print(std::ostream& os) const;

    };

    // main input types : directory entry, parameter entry
    struct parameter_entry {
      unsigned pointer;
      unsigned lines;
      std::string data;
    };

    struct loop_type {
      unsigned model_curve;
      unsigned orientation;  // 1=no reverse, 2=reverse
      std::vector<unsigned> curves;
    };

    struct composite_curve { // 102
      std::vector<unsigned> curves;
    };

    struct boundary_type { // 141
      unsigned type; // 0=model space only, 1=model and domain space
      unsigned preferred_trim_type; // 0=unspecified, 1=domain, 2=object space, 3=both
      unsigned surface;
      unsigned nloops;

      std::vector<loop_type> loops;
    };

    struct curve_on_surface { // 142
      unsigned creation_type; // 0=unspecified, 1=projection, 2=intersection, 3=isoparamteric curve
      unsigned surface;
      unsigned domain_curve;
      unsigned model_curve;
      unsigned preferred_trim_type; // 0=unspecified, 1=domain, 2=object space, 3=both
    };

    struct group_associativity { // 402
      std::vector<unsigned> entities;
    };

    struct bounded_surface { // 143
      unsigned              type; // 0=model space only, 1=model and domain space
      unsigned              surface;
      std::vector<unsigned> boundaries;
      directory_entry       info;
    };

    struct trimmed_surface { // 144
      unsigned              surface;
      unsigned              domain_is_outer_boundary;
      unsigned              outer_loop;
      std::vector<unsigned> inner_loops;
      directory_entry       info;
    };

    struct color {
      double r;
      double g;
      double b;
      std::string name;
    };

    struct point {
      double x;
      double y;
      double z;
      unsigned subfigure;
    };

  }


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

  std::vector<nurbsobject_ptr> load(std::string const& file, bool normalize_domain = true);
  
private: // methods

  bool                _load(std::fstream& is);
  bool                _parse_line(std::string& line_str, std::size_t line_nr);

  bool                _parse_directory(std::string const& line, unsigned section_index);
  bool                _parse_parameter(std::string const& line, unsigned section_index);

  iges::section_type  _get_section_type (char v) const;
  std::vector<std::string> _tokenize_parameters (std::string) const;

  bool                _create_entities();
  bool                _create_directory_entry(unsigned section_index);

  void                _create_circular_arc_entity(iges::directory_entry const&); // 100
  void                _create_composite_curve_entity(iges::directory_entry const&); // 102
  void                _create_plane_entity(iges::directory_entry const&); // 108
  void                _create_line_entity(iges::directory_entry const&); // 110
  void                _create_point_entity(iges::directory_entry const&); // 116
  void                _create_ruled_surface_entity(iges::directory_entry const&); // 118
  void                _create_matrix_entity(iges::directory_entry const&); // 124
  void                _create_rational_curve_entity(iges::directory_entry const&); // 126
  void                _create_rational_surface_entity(iges::directory_entry const&); // 128
  void                _create_boundary_entity(iges::directory_entry const&); // 141
  void                _create_curve_on_surface_entity(iges::directory_entry const&); // 142
  void                _create_bounded_surface_entity(iges::directory_entry const&); // 143
  void                _create_trimmed_surface_entity(iges::directory_entry const&); // 144
  void                _create_manifold_solid_brep_object_entity(iges::directory_entry const&); // 186
  void                _create_singular_subfigure_entity(iges::directory_entry const&); // 308
  void                _create_color_entity(iges::directory_entry const&); // 314
  void                _create_group_associativity_entity(iges::directory_entry const&); // 402
  void                _create_property_entity(iges::directory_entry const&); // 406
  void                _create_singular_subfigure_definition_entity(iges::directory_entry const&); // 408
  void                _create_vertex_list_entity(iges::directory_entry const&); // 502
  void                _create_edge_list_entity(iges::directory_entry const&); // 504
  void                _create_loop_entity(iges::directory_entry const&); // 508
  void                _create_face_entity(iges::directory_entry const&); // 510
  void                _create_closed_shell_entity(iges::directory_entry const&); // 514
  

  void                _create_objects();

  // helper methods
  void                _apply_trim_loop(unsigned section_index, unsigned surface_index, gpucast::nurbssurface& ns);

  math::nurbscurve2d _migrate_curve(math::nurbscurve3d const&);

private : // members

  std::string                                   _error;
  bool                                          _terminate = false;
  std::vector<nurbsobject_ptr>                  _result;

  // raw line input and mapping
  std::vector<std::string>                      _directory_entry_line1;
  std::vector<std::string>                      _directory_entry_line2;
  std::map<unsigned, unsigned>                  _first_section_id_map; // parameter entry identifier -> first section index

  // daata and parameter entries
  std::map<unsigned, std::string>               _parameter_data_map;   // first section id -> data
  std::vector<iges::directory_entry>            _directory_entries;

  // data entries
  std::map<unsigned, iges::directory_entry>       _surface_info_map;
  std::map<unsigned, iges::curve_on_surface>      _curve_on_surface_map;
  std::map<unsigned, iges::composite_curve>       _composite_curve_map;
  std::map<unsigned, iges::boundary_type>         _boundary_map;
  std::map<unsigned, iges::trimmed_surface>       _trimmed_surface_map;
  std::map<unsigned, iges::bounded_surface>       _bounded_surface_map;
  std::map<unsigned, iges::color>                 _color_map;
  std::map<unsigned, iges::group_associativity>   _group_map;
  std::map<unsigned, iges::point>                 _point_map;
  std::map<unsigned, math::matrix4d>              _matrix_map;
                                                  
  // nurbs data                                   
  std::map<unsigned, math::nurbscurve3d>          _curve_map;
  std::map<unsigned, math::nurbssurface3d>        _surface_map;
};

} // namespace gpucast

#endif // GPUCAST_IGS_LOADER_HPP
