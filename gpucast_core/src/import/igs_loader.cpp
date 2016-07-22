#include "gpucast/core/import/igs_loader.hpp"

#include <boost/algorithm/string.hpp>

#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>

#include <gpucast/core/nurbssurfaceobject.hpp>
#include <gpucast/math/matrix4x4.hpp>
#include <regex>

// header, system
#include <fstream>
#include <iostream>

namespace gpucast {

  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  namespace iges {

    void directory_entry::print(std::ostream& os) const {
      os << "section_index : " << section_index << std::endl;
      os << "entity_type : " << entity_type << std::endl;
      os << "parameter_data : " << parameter_data << std::endl;
      os << "line_font_pattern : " << line_font_pattern << std::endl;
      os << "level : " << level << std::endl;
      os << "view : " << view << std::endl;
      os << "transform : " << transform << std::endl;
      os << "label : " << label << std::endl;
      os << "status : " << status << std::endl;
      os << "line_weight : " << line_weight << std::endl;
      os << "color_number : " << color_number << std::endl;
      os << "parameter_line_count : " << parameter_line_count << std::endl;
      os << "form_number : " << form_number << std::endl;
      os << "entity_label : " << entity_label << std::endl;
      os << "entity_subscript : " << entity_subscript << std::endl;
    }
  }

  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  std::vector<std::shared_ptr<nurbssurfaceobject>>
  igs_loader::load(std::string const& file, bool normalize)
  {
    boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::warning);

    try {
      // try to open file
      std::fstream fstr;
      BOOST_LOG_TRIVIAL(info) << "Opening file : " << file << std::endl;
      fstr.open(file.c_str(), std::ios::in);

      // if file good parse stream
      if (fstr.good())
      {
        if (!_load(fstr))
        {
          throw std::runtime_error("igs_loader::load(): Failed to parse file: " + file);
        }
        fstr.close();
      }
      else {
        throw std::runtime_error("igs_loader::load(): Could not open file: " + file);
      }

      // normalize 
      if (normalize) {
        for (auto o : _result) {
          BOOST_LOG_TRIVIAL(error) << "to implement normalization of nurbssurfaceobjects." << std::endl;
        }
      }
    }
    catch (std::exception& e) {
      BOOST_LOG_TRIVIAL(error) << e.what() << std::endl;
      _result.clear();
    }

    return _result;
  }

  ////////////////////////////////////////////////////////////////////////////////
  bool
  igs_loader::_load(std::fstream& istr)
  {
    std::size_t line_nr = 0;
   
    while (istr)
    {
      std::string line;
      std::getline(istr, line);
      
      if (!_parse_line(line, line_nr++)) {
        throw std::runtime_error("igs_loader::load(): Failed to parse line " + std::to_string(line_nr));
      }
    }

    return (_error.length() > 1) ? false : true;
  }

  ////////////////////////////////////////////////////////////////////////////////
  bool igs_loader::_parse_line(std::string& line, std::size_t line_nr)
  {
    if (line.length() != iges::LINE_LENGTH) {
      if (_terminate) {
        return true;
      }
      else {
        BOOST_LOG_TRIVIAL(warning) << "Warning: Skipping line. Wrong length. Expected 80 characters in line " << line_nr << " : " << line << std::endl;
      }
    }
    else {
      // clip last 7 chars for section index
      unsigned index_offset = iges::LINE_LENGTH - iges::SECTION_INDEX_DIGITS;
      std::string section_index_string(line.begin() + index_offset, line.end());
      line.erase(line.begin() + index_offset, line.end());
      std::size_t section_index = std::stoll(section_index_string);
      
      // clip section type
      char sec_char = line.back();
      line.pop_back();
      auto section_type = _get_section_type(sec_char);

      switch (section_type) {
      case iges::section_flag:
        throw std::runtime_error("ASCII/Binary flag not supported.");
      case iges::section_start:
        // ignore start section
        break;
      case iges::section_global:
        // ignore global section
        break;
      case iges::section_directory:
        _parse_directory(line, section_index);
        break;
      case iges::section_parameter:
        _parse_parameter(line, section_index);
        break;
      case iges::section_terminate:
        // ignore terminate section
        return _create_entities();
      };
      
    }
    return true;
  }

  ////////////////////////////////////////////////////////////////////////////////
  bool igs_loader::_parse_directory(std::string const& line, unsigned section_index)
  {
    std::vector<std::string> directory_line(iges::DIRECTORY_COLUMN_COUNT);

    auto column_begin = line.begin();
    auto column_end = line.begin() + iges::DIRECTORY_COLUMN_WIDTH;

    for (unsigned i = 0; i != iges::DIRECTORY_COLUMN_COUNT; ++i) {
      directory_line[i] = std::string(column_begin, column_end);
      if (i + 1 != iges::DIRECTORY_COLUMN_COUNT) {
        std::advance(column_begin, iges::DIRECTORY_COLUMN_WIDTH);
        std::advance(column_end, iges::DIRECTORY_COLUMN_WIDTH);
      }
    }

    if (_directory_entry_line1.empty()) {
      _directory_entry_line1 = directory_line;
    }
    else {
      _directory_entry_line2 = directory_line;
      _create_directory_entry(section_index-1);
      
      // clear stack
      _directory_entry_line1.clear();
      _directory_entry_line2.clear();
    }

    return true;
  }

  ////////////////////////////////////////////////////////////////////////////////
  bool igs_loader::_parse_parameter(std::string const& line, unsigned section_index)
  {
    std::string data(line.begin(), line.begin() + 64);
    std::string pointer_str(line.begin() + 65, line.begin() + 72);

    unsigned param_pointer = iges::stoi(pointer_str);

    // store relation between parameter id and section id
    if (!_first_section_id_map.count(param_pointer)) {
      _first_section_id_map[param_pointer] = section_index;
      _parameter_data_map.insert(std::make_pair(section_index, ""));
    }

    unsigned current_id = _first_section_id_map[param_pointer];

    if (!_parameter_data_map.count(current_id)) {
      BOOST_LOG_TRIVIAL(error) << "igs_loader::_parse_parameter(): Invalid parameter entry.\n";
    }
    _parameter_data_map.at(current_id).append(data);

    return true;
  }

  ////////////////////////////////////////////////////////////////////////////////
  iges::section_type igs_loader::_get_section_type(char v) const
  {
    switch (v) {
    case 'F': return iges::section_flag;
    case 'S': return iges::section_start;
    case 'G': return iges::section_global;
    case 'D': return iges::section_directory;
    case 'P': return iges::section_parameter;
    case 'T': return iges::section_terminate;
    default: throw std::runtime_error("igs_loader::_get_scetion_type(): Unknwon section type:" + v);
    };
  }

  ////////////////////////////////////////////////////////////////////////////////
  std::vector<std::string> igs_loader::_tokenize_parameters(std::string s) const
  {
    s.erase(std::remove_if(s.begin(), s.end(), isspace), s.end());
    std::vector<std::string> tokenized_data;

    std::regex ws_or_comma("[,;]");
    std::copy(std::sregex_token_iterator(s.begin(), s.end(), ws_or_comma, -1),
      std::sregex_token_iterator(),
      std::back_inserter(tokenized_data));

    return tokenized_data;
  }


  ////////////////////////////////////////////////////////////////////////////////
  bool igs_loader::_create_directory_entry(unsigned section_index)
  {
    try {
      iges::directory_entry entry;

      entry.section_index = section_index;

      entry.entity_type = iges::stoi(_directory_entry_line1[0]);         // 1:01-08
      entry.parameter_data = iges::stoi(_directory_entry_line1[1]);      // 1:09-16
      entry.structure = iges::stoi(_directory_entry_line1[2]);           // 1:17-24
      entry.line_font_pattern = iges::stoi(_directory_entry_line1[3]);   // 1:25-32
      entry.level = iges::stoi(_directory_entry_line1[4]);               // 1:33-40
      entry.view = iges::stoi(_directory_entry_line1[5]);                // 1:41-48
      entry.transform = iges::stoi(_directory_entry_line1[6]);           // 1:49-56
      entry.label = iges::stoi(_directory_entry_line1[7]);               // 1:57-64
      entry.status = iges::stoi(_directory_entry_line1[8]);              // 1:65-72

      entry.line_weight = iges::stoi(_directory_entry_line2[1]);         // 2:09-16
      entry.color_number = std::abs(iges::stoi(_directory_entry_line2[2]));        // 2:17-24
      entry.parameter_line_count = iges::stoi(_directory_entry_line2[3]);// 2:25-32
      entry.form_number = iges::stoi(_directory_entry_line2[4]);         // 2:33-40
      entry.entity_label = _directory_entry_line2[7];                   // 2:57-64
      entry.entity_subscript = iges::stoi(_directory_entry_line2[8]);    // 2:65-72

      _directory_entries.push_back(entry);
    }
    catch (std::exception& e) {
      throw std::runtime_error(std::string("igs_loader::_create_directory_entry(): Failed to create directory entry. Section index : ") + std::to_string(section_index) + e.what());
    }
    return true;
  }

  ////////////////////////////////////////////////////////////////////////////////
  bool igs_loader::_create_entities() {

    for (auto const& e : _directory_entries) {

      try {
        switch (e.entity_type) {
        case 100: _create_circular_arc_entity(e);  break;
        case 102: _create_composite_curve_entity(e);  break;
        case 108: _create_plane_entity(e);  break;
        case 110: _create_line_entity(e);  break;
        case 118: _create_ruled_surface_entity(e);  break;
        case 124: _create_matrix_entity(e); break;
        case 126: _create_rational_curve_entity(e); break;
        case 128: _create_rational_surface_entity(e); break;
        case 141: _create_boundary_entity(e); break;
        case 142: _create_curve_on_surface_entity(e); break;
        case 143: _create_bounded_surface_entity(e); break;
        case 144: _create_trimmed_surface_entity(e); break;
        case 186: _create_manifold_solid_brep_object_entity(e); break;
        case 308: _create_singular_subfigure_entity(e); break;
        case 314: _create_color_entity(e); break;
        case 402: _create_group_associativity_entity(e); break;
        case 406: _create_property_entity(e); break;
        case 408: _create_singular_subfigure_definition_entity(e); break;
        case 502: _create_vertex_list_entity(e); break;
        case 504: _create_edge_list_entity(e); break;
        case 508: _create_loop_entity(e); break;
        case 510: _create_face_entity(e); break;
        case 514: _create_closed_shell_entity(e); break;
          
        default:
          throw std::runtime_error( std::string("igs_loader::_create_entities() : Error, unknown entity ") + std::to_string(e.entity_type));
        }
      } catch (std::exception& e) {
        BOOST_LOG_TRIVIAL(error) << "igs_loader::_create_entities(). Warning. " << e.what() << "\n" << "Trying to continue." << std::endl;
      }
    }
    _terminate = true;

    _create_objects();

    return true;
  }

  ////////////////////////////////////////////////////////////////////////////////
  void igs_loader::_create_circular_arc_entity(iges::directory_entry const& de) // 100
  {
    try {
      auto tokenized_data = _tokenize_parameters(_parameter_data_map[de.parameter_data]);

      unsigned index = 0;
      unsigned entity_type = iges::stoi(tokenized_data[index++]);

      double zt = iges::stod(tokenized_data[index++]);

      double center_x = iges::stod(tokenized_data[index++]);
      double center_y = iges::stod(tokenized_data[index++]);

      double start_x = iges::stod(tokenized_data[index++]);
      double start_y = iges::stod(tokenized_data[index++]);

      double end_x = iges::stod(tokenized_data[index++]);
      double end_y = iges::stod(tokenized_data[index++]);
      
      std::vector<math::point3d> points;
      math::point3d p_center{ center_x, center_y, zt };
      math::point3d p_start{ start_x, start_y, zt };
      math::point3d p_end{ end_x, end_y, zt };

      auto ra = p_start - p_center;
      auto rb = p_end - p_center;

      ra.normalize();
      rb.normalize();

      auto alpha = math::dot(ra, rb);
      auto angle_degree = std::acos(alpha);

      auto p1 = math::make_rotation_z( (1.0  /3.0) * (angle_degree / (2 * M_PI)));
      auto p2 = math::make_rotation_z( (2.0 / 3.0) * (angle_degree / (2 * M_PI)));

      points.push_back(p_start);
      points.push_back(p1);
      points.push_back(p2);
      points.push_back(p_end);

      std::vector<double> kv{0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0};
      
      math::nurbscurve3d nc;
      nc.degree(3);
      nc.set_points(points.begin(), points.end());
      nc.set_knotvector(kv.begin(), kv.end());

      _curve_map.insert({de.section_index, nc});

      BOOST_LOG_TRIVIAL(warning) << "igs_loader::_create_circular_arc_entity() : Warning. Please verify convert from circular arc to NURBS curve.\n";

    }
    catch (std::exception& e) {
      throw std::runtime_error(std::string("igs_loader::_create_circular_arc_entity():") + e.what());
    }
  }

  ////////////////////////////////////////////////////////////////////////////////
  void igs_loader::_create_composite_curve_entity(iges::directory_entry const& de) // 102
  {
    try {
      auto tokenized_data = _tokenize_parameters(_parameter_data_map[de.parameter_data]);

      unsigned index = 0;
      unsigned entity_type = iges::stoi(tokenized_data[index++]);

      unsigned ncurves = iges::stoi(tokenized_data[index++]);
      iges::composite_curve composite;

      for (unsigned i = 0; i != ncurves; ++i) {
        unsigned curve_index = iges::stoi(tokenized_data[index++]);
        composite.curves.push_back(curve_index);
      }

      _composite_curve_map.insert(std::make_pair(de.section_index, composite));
    }
    catch (std::exception& e) {
      throw std::runtime_error(std::string("igs_loader::_create_composite_curve_entity():") + e.what());
    }
  }

  ////////////////////////////////////////////////////////////////////////////////
  void igs_loader::_create_plane_entity(iges::directory_entry const& de) // 108
  {
    BOOST_LOG_TRIVIAL(warning) << "Missing implementation. Type 108.\n";
  }

  ////////////////////////////////////////////////////////////////////////////////
  void igs_loader::_create_line_entity(iges::directory_entry const& de) // 110
  {
    try {
      auto tokenized_data = _tokenize_parameters(_parameter_data_map[de.parameter_data]);

      unsigned index = 0;
      unsigned entity_type = iges::stoi(tokenized_data[index++]);

      // read general information
      double x0 = iges::stod(tokenized_data[index++]);
      double y0 = iges::stod(tokenized_data[index++]);
      double z0 = iges::stod(tokenized_data[index++]);
      double x1 = iges::stod(tokenized_data[index++]);
      double y1 = iges::stod(tokenized_data[index++]);
      double z1 = iges::stod(tokenized_data[index++]);

      // create control point array
      std::vector<math::point3d> points;
      points.push_back(math::point3d(x0, y0, z0));
      points.push_back(math::point3d(x1, y1, z1));
      points[0].weight(1.0);
      points[1].weight(1.0);

      // set nurbs curve as line with according parameters
      math::nurbscurve3d nc;

      nc.degree(1); // linear
      std::vector<double> knots = { 0.0, 0.0, 1.0, 1.0 };
      nc.set_knotvector(knots.begin(), knots.end());
      nc.set_points(points.begin(), points.end());

      _curve_map.insert({ de.section_index, nc });
    }
    catch (std::exception& e) {
      throw std::runtime_error(std::string("igs_loader::_create_line_entity():") + e.what());
    }
  }
  ////////////////////////////////////////////////////////////////////////////////
  void igs_loader::_create_point_entity(iges::directory_entry const& de) // 116
  {
    try {
      auto tokenized_data = _tokenize_parameters(_parameter_data_map[de.parameter_data]);

      unsigned index = 0;
      unsigned entity_type = iges::stoi(tokenized_data[index++]);

      iges::point p;

      p.x = iges::stod(tokenized_data[index++]);
      p.y = iges::stod(tokenized_data[index++]);
      p.z = iges::stod(tokenized_data[index++]);

      p.subfigure = iges::stoi(tokenized_data[index++]);

      _point_map.insert({ de.section_index, p });
    }
    catch (std::exception& e) {
      throw std::runtime_error(std::string("igs_loader::_create_point_entity():") + e.what());
    }
  }


  ////////////////////////////////////////////////////////////////////////////////
  void igs_loader::_create_ruled_surface_entity(iges::directory_entry const& de) // 118
  {
    BOOST_LOG_TRIVIAL(warning) << "Missing implementation. Type 118.\n";
  }

  ////////////////////////////////////////////////////////////////////////////////
  void igs_loader::_create_matrix_entity(iges::directory_entry const& de) // 124
  {
    try {
      auto tokenized_data = _tokenize_parameters(_parameter_data_map[de.parameter_data]);

      unsigned index = 0;
      unsigned entity_type = iges::stoi(tokenized_data[index++]);

      math::matrix4d m;

      switch (de.form_number) {
      case 0: break;
      case 1: break;
      case 10: break;
      case 11: break;
      case 12: break;
        
        // first row
        m[0] = iges::stod(tokenized_data[index++]);
        m[4] = iges::stod(tokenized_data[index++]);
        m[8] = iges::stod(tokenized_data[index++]);
        m[12] = iges::stod(tokenized_data[index++]); // translation x
        // second row
        m[1] = iges::stod(tokenized_data[index++]);
        m[5] = iges::stod(tokenized_data[index++]);
        m[9] = iges::stod(tokenized_data[index++]);
        m[13] = iges::stod(tokenized_data[index++]); // translation y
        // third row
        m[2] = iges::stod(tokenized_data[index++]);
        m[6] = iges::stod(tokenized_data[index++]);
        m[10] = iges::stod(tokenized_data[index++]);
        m[14] = iges::stod(tokenized_data[index++]); // translation z
        // fourth row
        // m[3] = 0.0;
        // m[7] = 0.0;
        // m[11] = 0.0;
        // m[15] = 1.0;
        _matrix_map.insert({ de.section_index, m });
        BOOST_LOG_TRIVIAL(error) << "Warning: Unvarified matrix input. Todo check orientation and translation.\n";
        break;
      default:
        throw std::runtime_error("igs_loader::_create_matrix_entity() : Unhandled matrix type.");
      }

    }
    catch (std::exception& e) {
      throw std::runtime_error(std::string("igs_loader::_create_rational_curve_entity():") + e.what());
    }
  }

  ////////////////////////////////////////////////////////////////////////////////
  void igs_loader::_create_rational_curve_entity(iges::directory_entry const& de) // 126
  {
    try {
      auto tokenized_data = _tokenize_parameters(_parameter_data_map[de.parameter_data]);

      unsigned index = 0;
      unsigned entity_type = iges::stoi(tokenized_data[index++]);

      // read general information
      unsigned upper_index_sum = iges::stoi(tokenized_data[index++]);
      unsigned degree = iges::stoi(tokenized_data[index++]);
      bool is_planar = (1 == iges::stoi(tokenized_data[index++]));
      bool is_closed = (1 == iges::stoi(tokenized_data[index++]));
      bool is_rational = (0 == iges::stoi(tokenized_data[index++]));
      bool is_periodic = (0 == iges::stoi(tokenized_data[index++]));

      unsigned npoints = 1 + upper_index_sum;
      unsigned order = degree + 1;
      unsigned nknots = npoints + order;

      const unsigned knot_base_id = index++;
      const unsigned weight_base_id = knot_base_id + nknots;
      const unsigned point_base_id = weight_base_id + npoints;

      // read knots
      std::vector<double> knots;
      for (unsigned i = knot_base_id; i != knot_base_id + nknots; ++i) {
        auto knot = iges::stod(tokenized_data[i]);
        knots.push_back(knot);
      }

      // read weights
      std::vector<double> weights;
      for (unsigned i = weight_base_id; i != weight_base_id + npoints; ++i) {
        auto weight = iges::stod(tokenized_data[i]);
        weights.push_back(weight);
      }

      std::vector<math::point3d> points(npoints);

      // read control points
      int j = 0;
      for (unsigned i = point_base_id; i != point_base_id + 3 * npoints; ++i, ++j) {
        unsigned pid = j / 3;
        unsigned coord = j % 3;
        points[pid][coord] = iges::stod(tokenized_data[i]);
        points[pid].weight(weights[pid]);
      }

      // create curve
      math::nurbscurve3d nc;
      nc.degree(degree);
      nc.set_knotvector(knots.begin(), knots.end());
      nc.set_points(points.begin(), points.end());

      _curve_map.insert({ de.section_index, nc });
    }
    catch (std::exception& e) {
      throw std::runtime_error(std::string("igs_loader::_create_rational_curve_entity():") + e.what());
    }
    
  }

  ////////////////////////////////////////////////////////////////////////////////
  void igs_loader::_create_rational_surface_entity(iges::directory_entry const& de) // 128
  {
#if 0
    if (de.form_number != 0) {
      switch (de.form_number) {
      case 1 : throw std::runtime_error(std::string("igs_loader::_create_rational_surface_entity(): Unhandled surface representation: plane, form number ") + std::to_string(de.form_number));
      case 2 : throw std::runtime_error(std::string("igs_loader::_create_rational_surface_entity(): Unhandled surface representation: right circular cylinder, form number ") + std::to_string(de.form_number));
      case 3 : throw std::runtime_error(std::string("igs_loader::_create_rational_surface_entity(): Unhandled surface representation: cone, form number ") + std::to_string(de.form_number)); 
      case 4 : throw std::runtime_error(std::string("igs_loader::_create_rational_surface_entity(): Unhandled surface representation: sphere, form number ") + std::to_string(de.form_number)); 
      case 5 : throw std::runtime_error(std::string("igs_loader::_create_rational_surface_entity(): Unhandled surface representation: torus, form number ") + std::to_string(de.form_number));
      case 6 : throw std::runtime_error(std::string("igs_loader::_create_rational_surface_entity(): Unhandled surface representation: surface of revolution, form number ") + std::to_string(de.form_number));
      case 7 : throw std::runtime_error(std::string("igs_loader::_create_rational_surface_entity(): Unhandled surface representation: tabulated cylinder, form number ") + std::to_string(de.form_number));
      case 8 : throw std::runtime_error(std::string("igs_loader::_create_rational_surface_entity(): Unhandled surface representation: ruled surface, form number ") + std::to_string(de.form_number));
      case 9 : throw std::runtime_error(std::string("igs_loader::_create_rational_surface_entity(): Unhandled surface representation: general quadric surface, form number ") + std::to_string(de.form_number));
      default : throw std::runtime_error(std::string("igs_loader::_create_rational_surface_entity(): Unhandled surface representation: form number ") + std::to_string(de.form_number));
      };
    }
#endif
    try {
      auto tokenized_data = _tokenize_parameters(_parameter_data_map[de.parameter_data]);

      unsigned index = 0;
      unsigned entity_type = iges::stoi(tokenized_data[index++]);

      // size of control point mesh
      unsigned npoints_u = iges::stoi(tokenized_data[index++]) + 1;
      unsigned npoints_v = iges::stoi(tokenized_data[index++]) + 1;

      // degree u,v
      unsigned degree_u = iges::stoi(tokenized_data[index++]);
      unsigned degree_v = iges::stoi(tokenized_data[index++]);

      // properties
      bool is_closed_u = (0 == iges::stoi(tokenized_data[index++]));
      bool is_closed_v = (0 == iges::stoi(tokenized_data[index++]));
      bool is_rational = (0 == iges::stoi(tokenized_data[index++]));
      bool is_periodic_u = (1 == iges::stoi(tokenized_data[index++]));
      bool is_periodic_v = (1 == iges::stoi(tokenized_data[index++]));

      // knot sequences
      unsigned order_u = degree_u + 1;
      unsigned order_v = degree_v + 1;

      unsigned nknots_u = npoints_u + order_u;
      unsigned nknots_v = npoints_v + order_v;
      unsigned npoints = npoints_u * npoints_v;

      const unsigned knot_base_id = index;
      const unsigned weight_base_id = knot_base_id + nknots_u + nknots_v;
      const unsigned point_base_id = weight_base_id + npoints;

      // read knots 
      std::vector<double> knots_u;
      std::vector<double> knots_v;
      for (unsigned i = knot_base_id; i != knot_base_id + nknots_u; ++i) {
        knots_u.push_back(iges::stod(tokenized_data[i]));
      }

      for (unsigned i = knot_base_id + nknots_u; i != knot_base_id + nknots_u + nknots_v; ++i) {
        knots_v.push_back(iges::stod(tokenized_data[i]));
      }

      // read weights
      std::vector<double> weights;
      for (unsigned i = weight_base_id; i != weight_base_id + npoints; ++i) {
        weights.push_back(iges::stod(tokenized_data[i]));
      }

      // read points
      std::vector<math::point3d> points(npoints);
      int j = 0;
      for (unsigned i = point_base_id; i != point_base_id + 3 * npoints; ++i, ++j) {
        unsigned pid = j / 3;
        unsigned coord = j % 3;
        points[pid][coord] = iges::stod(tokenized_data[i]);
        points[pid].weight(weights[pid]);
      }

      // create surface
      math::nurbssurface3d ns;
      ns.numberofpoints_u(npoints_u);
      ns.numberofpoints_v(npoints_v);
      ns.degree_u(degree_u);
      ns.degree_v(degree_v);
      ns.knotvector_u(knots_u.begin(), knots_u.end());
      ns.knotvector_v(knots_v.begin(), knots_v.end());
      ns.set_points(points.begin(), points.end());

      _surface_map.insert({ de.section_index, ns });
      _surface_info_map.insert({de.section_index, de});
    }
    catch (std::exception& e) {
      throw std::runtime_error(std::string("igs_loader::_create_rational_surface_entity():") + e.what());
    }
    
  }

  ////////////////////////////////////////////////////////////////////////////////
  void igs_loader::_create_boundary_entity(iges::directory_entry const& de) // Type 141
  {
    try {
      auto tokenized_data = _tokenize_parameters(_parameter_data_map[de.parameter_data]);

      unsigned index = 0;
      unsigned entity_type = iges::stoi(tokenized_data[index++]);
      iges::boundary_type boundary;

      boundary.type = iges::stoi(tokenized_data[index++]);
      boundary.preferred_trim_type = iges::stoi(tokenized_data[index++]);
      boundary.surface = iges::stoi(tokenized_data[index++]);
      boundary.nloops = iges::stoi(tokenized_data[index++]); 

      unsigned unprocessed_loops = boundary.nloops;

      while (unprocessed_loops-- > 0) {
        iges::loop_type loop;
        loop.model_curve = iges::stoi(tokenized_data[index++]);
        loop.orientation = iges::stoi(tokenized_data[index++]);
        
        unsigned ncurves = iges::stoi(tokenized_data[index++]);
        for (int i = 0; i != ncurves; ++i) {
          loop.curves.push_back(iges::stoi(tokenized_data[index++]));
        }

        boundary.loops.push_back(loop);
      }

      _boundary_map.insert({de.section_index, boundary });
      BOOST_LOG_TRIVIAL(debug) << "Found Boundary (Type 141)." << "Internal pointer: " << de.parameter_data << "\n";
    }
    catch (std::exception& e) {
      throw std::runtime_error(std::string("igs_loader::_create_boundary_entity():") + e.what());
    }
  }

  ////////////////////////////////////////////////////////////////////////////////
  void igs_loader::_create_curve_on_surface_entity(iges::directory_entry const& de) // Type 142
  {
    try {
      auto tokenized_data = _tokenize_parameters(_parameter_data_map[de.parameter_data]);

      unsigned index = 0;
      unsigned entity_type = iges::stoi(tokenized_data[index++]);

      iges::curve_on_surface curve_on_face;

      curve_on_face.creation_type = iges::stoi(tokenized_data[index++]);
      curve_on_face.surface = iges::stoi(tokenized_data[index++]);
      curve_on_face.domain_curve = iges::stoi(tokenized_data[index++]); // D
      curve_on_face.model_curve = iges::stoi(tokenized_data[index++]); // C
      curve_on_face.preferred_trim_type = iges::stoi(tokenized_data[index++]);

      switch (curve_on_face.preferred_trim_type) {
      case 0 : // unspecified
        break;
      case 1 : // trim in domain : S x D
        break;
      case 2 : // trim in object space : C
        throw std::runtime_error("igs_loader::_create_curve_on_surface_entity(): Trim in object space not supported.");
        break;
      case 3 : // trim either way: C or S x B
        break;
      default :
        throw std::runtime_error("igs_loader::_create_curve_on_surface_entity(): Invalid trim method.");
      };

      _curve_on_surface_map.insert({ de.section_index, curve_on_face });

      BOOST_LOG_TRIVIAL(debug) << "Found Curve on Surface (Type 142)." << "Internal pointer: " << de.parameter_data << "\n";
    }
    catch (std::exception& e) {
      throw std::runtime_error(std::string("igs_loader::_create_curve_on_surface_entity():") + e.what());
    }
  }

  ////////////////////////////////////////////////////////////////////////////////
  void igs_loader::_create_bounded_surface_entity(iges::directory_entry const& de) // Type 143
  {
    try {
      auto tokenized_data = _tokenize_parameters(_parameter_data_map[de.parameter_data]);

      unsigned index = 0;
      unsigned entity_type = iges::stoi(tokenized_data[index++]);

      iges::bounded_surface bsurface;
      bsurface.info = de;

      // read boundary information
      bsurface.type = iges::stoi(tokenized_data[index++]);
      if (bsurface.type == 0) {
        throw std::runtime_error("Modelspace trim curves not supported.");
      }

      bsurface.surface = iges::stoi(tokenized_data[index++]);
      unsigned nboundaries = iges::stoi(tokenized_data[index++]);

      for (unsigned i = index; i != index + nboundaries; ++i) {
        bsurface.boundaries.push_back(iges::stoi(tokenized_data[i]));
      }
      

      _bounded_surface_map.insert({ de.section_index, bsurface });

      BOOST_LOG_TRIVIAL(debug) << "Found Bounded Surface (Type 142)." << "Internal pointer: " << de.parameter_data << "\n";
    }
    catch (std::exception& e) {
      throw std::runtime_error(std::string("igs_loader::_create_bounded_surface_entity():") + e.what());
    }
  }

  ////////////////////////////////////////////////////////////////////////////////
  void igs_loader::_create_trimmed_surface_entity(iges::directory_entry const& de) // Type 144
  {
    try {
      auto tokenized_data = _tokenize_parameters(_parameter_data_map[de.parameter_data]);
      unsigned index = 0;
      unsigned entity_type = iges::stoi(tokenized_data[index++]);

      iges::trimmed_surface tsurface;
      tsurface.info = de;

      // read boundary information
      tsurface.surface = iges::stoi(tokenized_data[index++]);
      tsurface.domain_is_outer_boundary = iges::stoi(tokenized_data[index++]);
      unsigned n_inner_loops = iges::stoi(tokenized_data[index++]);
      tsurface.outer_loop = iges::stoi(tokenized_data[index++]);

      // gather loops
      for (unsigned i = index; i != index + n_inner_loops; ++i) {
        tsurface.inner_loops.push_back(iges::stoi(tokenized_data[i]));
      }

      _trimmed_surface_map.insert({ de.section_index, tsurface });

      BOOST_LOG_TRIVIAL(debug) << "Found Trimmed Surface (Type 144)." << "Internal pointer: " << de.parameter_data << "\n";
    }
    catch (std::exception& e) {
      throw std::runtime_error(std::string("igs_loader::_create_trimmed_surface_entity():") + e.what());
    }
  }

  ////////////////////////////////////////////////////////////////////////////////
  void igs_loader::_create_manifold_solid_brep_object_entity(iges::directory_entry const& de) // 186
  {
    BOOST_LOG_TRIVIAL(warning) << "Missing implementation. Type 186." << "Internal pointer: " << de.parameter_data << "\n";
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  void igs_loader::_create_singular_subfigure_entity(iges::directory_entry const& de) // 308
  {
    BOOST_LOG_TRIVIAL(warning) << "Missing implementation. Type 308." << "Internal pointer: " << de.parameter_data << "\n";
  }

  ////////////////////////////////////////////////////////////////////////////////
  void igs_loader::_create_color_entity(iges::directory_entry const& de) // 314
  {
    try {
      auto tokenized_data = _tokenize_parameters(_parameter_data_map[de.parameter_data]);
      unsigned index = 0;
      unsigned entity_type = iges::stoi(tokenized_data[index++]);

      iges::color c;

      // IGES stores intensities values between 0.0 and 100.0 
      c.r = iges::stod(tokenized_data[index++]) / iges::MAX_COLOR_INTENSITY;
      c.g = iges::stod(tokenized_data[index++]) / iges::MAX_COLOR_INTENSITY;
      c.b = iges::stod(tokenized_data[index++]) / iges::MAX_COLOR_INTENSITY;
      c.name = tokenized_data[index++];

      _color_map.insert({ de.section_index, c });

      BOOST_LOG_TRIVIAL(debug) << "Found Color (Type 314)." << "Internal pointer: " << de.parameter_data << "\n";
    }
    catch (std::exception& e) {
      throw std::runtime_error(std::string("igs_loader::_create_trimmed_surface_entity():") + e.what());
    }
  }

  ////////////////////////////////////////////////////////////////////////////////
  void igs_loader::_create_group_associativity_entity(iges::directory_entry const& de) // 402
  {
    BOOST_LOG_TRIVIAL(warning) << "Missing implementation. Type 402." << "Internal pointer: " << de.parameter_data << "\n";
  }

  ////////////////////////////////////////////////////////////////////////////////
  void igs_loader::_create_property_entity(iges::directory_entry const& de) // 406
  {
    BOOST_LOG_TRIVIAL(warning) << "Missing implementation. Type 406." << "Internal pointer: " << de.parameter_data << "\n";
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  void igs_loader::_create_singular_subfigure_definition_entity(iges::directory_entry const& de) // 408
  {
    BOOST_LOG_TRIVIAL(warning) << "Missing implementation. Type 408." << "Internal pointer: " << de.parameter_data << "\n";
  }

  ////////////////////////////////////////////////////////////////////////////////
  void igs_loader::_create_vertex_list_entity(iges::directory_entry const& de) // 502
  {
    BOOST_LOG_TRIVIAL(warning) << "Missing implementation. Type 502." << "Internal pointer: " << de.parameter_data << "\n";
  }

  ////////////////////////////////////////////////////////////////////////////////
  void igs_loader::_create_edge_list_entity(iges::directory_entry const& de) // 504
  {
    BOOST_LOG_TRIVIAL(warning) << "Missing implementation. Type 504." << "Internal pointer: " << de.parameter_data << "\n";
  }

  ////////////////////////////////////////////////////////////////////////////////
  void igs_loader::_create_loop_entity(iges::directory_entry const& de) // 508
  {
    BOOST_LOG_TRIVIAL(warning) << "Missing implementation. Type 508." << "Internal pointer: " << de.parameter_data << "\n";
  }

  ////////////////////////////////////////////////////////////////////////////////
  void igs_loader::_create_face_entity(iges::directory_entry const& de) // 510
  {
    BOOST_LOG_TRIVIAL(warning) << "Missing implementation. Type 510." << "Internal pointer: " << de.parameter_data << "\n";
  }

  ////////////////////////////////////////////////////////////////////////////////
  void igs_loader::_create_closed_shell_entity(iges::directory_entry const& de) // 514
  {
    BOOST_LOG_TRIVIAL(warning) << "Missing implementation. Type 514." << "Internal pointer: " << de.parameter_data << "\n";
  }

  ////////////////////////////////////////////////////////////////////////////////
  void igs_loader::_create_objects() {

    //for (auto i : _curve_on_surface_map) {
    //  std::cout << "curve_on_surface_map : " << i.first << std::endl;
    //}
    //
    //for (auto i : _boundary_map) {
    //  std::cout << "boundary : " << i.first << std::endl;
    //}

    //for (auto i : _surface_map) {
    //  std::cout << "surface : " << i.first << std::endl;
    //}
    BOOST_LOG_TRIVIAL(info) << "Finshed parsing file. Initializing objects...\n";

    std::map<int, nurbsobject_ptr> colored_faces;

    // keep track of untrimmed surfaces 
    auto untrimmed_surfaces = _surface_map;
    unsigned ntrimmed_faces = 0; 
    unsigned nbounded_faces = 0;

    // create bounded surfaces 
    for (auto i : _bounded_surface_map) {
      BOOST_LOG_TRIVIAL(debug) << "Creating trimmed surface. Index: " << i.first << ", COLOR = " << i.second.info.color_number << std::endl;
      auto surface_index = i.second.surface;
      if (_surface_map.count(surface_index)) {
        // create trimmed surface 
        auto surface = _surface_map.at(surface_index);
        gpucast::nurbssurface trimmed_ns(surface);
        ++nbounded_faces;

        // remove surface from surface stack
        untrimmed_surfaces.erase(surface_index);

        if (i.second.type == 0) {
          throw std::runtime_error("igs_loader::_create_objects() : Model space boundaries not supported.");
        }
        else {
          for (auto b : i.second.boundaries) {
            if (_boundary_map.count(b)) {
              for (auto l : _boundary_map.at(b).loops) {
                for (auto c : l.curves) {
                  if (_curve_map.count(c)) {
                    BOOST_LOG_TRIVIAL(debug) << "curve valid.\n" << std::endl;
                  }
                }
              }
            }
            else {
              BOOST_LOG_TRIVIAL(debug) << "No such boundary.\n" << std::endl;
            }
          }
        }
      }
    }


    // create trimmed surfaces 
    for (auto i : _trimmed_surface_map) { 
      BOOST_LOG_TRIVIAL(debug) << "Creating trimmed surface. Index: " << i.first << ", COLOR = " << i.second.info.color_number << std::endl;
      auto surface_index = i.second.surface; 
      auto color_index = i.second.info.color_number;
      if (_surface_map.count(surface_index)) {
        
        // create trimmed surface 
        auto surface = _surface_map.at(surface_index);
        gpucast::nurbssurface trimmed_ns(surface);
        ++ntrimmed_faces;

        // remove surface from surface stack
        untrimmed_surfaces.erase(surface_index);

        // adding outer and inner trim loops
        if (i.second.domain_is_outer_boundary) { // no explicit outer boundary given
          _apply_trim_loop(i.second.outer_loop, surface_index, trimmed_ns);   
          for (auto inner_loop_index : i.second.inner_loops) {
            _apply_trim_loop(inner_loop_index, surface_index, trimmed_ns);
          }
        }
        else { // custom outer boundary
          BOOST_LOG_TRIVIAL(error) << "igs_loader::_create_objects(). D is not an outer boundary. To implement.\n";
        }
        // add trimmed surface to result container
        
        if (!colored_faces.count(color_index)) {
          colored_faces.insert({ color_index, std::make_shared<nurbssurfaceobject>() });
        } 
        colored_faces.at(color_index)->add(trimmed_ns);

      }
      else {
        BOOST_LOG_TRIVIAL(error) << "igs_loader::_create_objects(): Missing surface. Section index : " << surface_index << ".\n";
      }
    }

    // create untrimmed surfaces 
    for (auto untrimmed_nurbs : untrimmed_surfaces) {
      gpucast::nurbssurface untrimmed_ns(untrimmed_nurbs.second);

      auto color_index = _surface_info_map.at(untrimmed_nurbs.first).color_number;

      if (!colored_faces.count(color_index)) {
        colored_faces.insert({ color_index, std::make_shared<nurbssurfaceobject>() });
      }
      colored_faces.at(color_index)->add(untrimmed_ns);
    }

    for (auto color_face : colored_faces) 
    {
      auto object_ptr = color_face.second;
      // apply color to object
      if (_color_map.count(color_face.first)) {
        auto color = _color_map.at(color_face.first);
        object_ptr->color(math::vec3f(color.r, color.g, color.b));
      }
      else {
        BOOST_LOG_TRIVIAL(error) << "igs_loader::_create_objects(): Missing color. Index : " << color_face.first << ".\n";
      }
      // push nurbsobject to  result
      _result.push_back(object_ptr);
    }

    BOOST_LOG_TRIVIAL(info) << "Finshed creating objects. " << ntrimmed_faces << " trimmed surfaces. " << nbounded_faces << " bounded surfaces. " << untrimmed_surfaces.size() << "untrimmed surfaces." << std::endl;
  }

  ////////////////////////////////////////////////////////////////////////////////
  void igs_loader::_apply_trim_loop(unsigned section_index, 
                                     unsigned surface_index,
                                     gpucast::nurbssurface& trimmed_surface)
  {
    //std::cout << "Applying trim to surface : " << surface_index << " loop index : " << section_index << "\n";
    if (_curve_on_surface_map.count(section_index)) {

      auto loop = _curve_on_surface_map.at(section_index);

      // only handle domain trimming
      if (loop.preferred_trim_type == 1) {

        // add single curve as trim loop
        if (_curve_map.count(loop.domain_curve)) {

          auto trim_curve = _curve_map.at(loop.domain_curve);
          gpucast::nurbssurface::curve_container single_curve_trim_loop = { _migrate_curve(trim_curve) };
          trimmed_surface.add(single_curve_trim_loop);
        }
        else { // add curve composite as trim loop
          if (_composite_curve_map.count(loop.domain_curve)) {
            
            auto trim_composite = _composite_curve_map.at(loop.domain_curve);
            gpucast::nurbssurface::curve_container composite_trim_loop;
            for (auto curve_index : trim_composite.curves) {
              if (_curve_map.count(curve_index)) {
                auto curve = _curve_map.at(curve_index);
                auto curve2d = _migrate_curve(curve);

                if (!curve2d.verify()) {
                  BOOST_LOG_TRIVIAL(error) << "igs_loader::_apply_trim_loop(): Curve migration failed. Section index :  " << curve_index << ".\n";
                }
                composite_trim_loop.push_back(curve2d);
              }
              else {
                BOOST_LOG_TRIVIAL(error) << "igs_loader::_apply_trim_loop(): Missing curve. Section index :  " << curve_index << ".\n";
              }
            }
            trimmed_surface.add(composite_trim_loop);
          }
          else {
            BOOST_LOG_TRIVIAL(error) << "igs_loader::_apply_trim_loop(): Missing curve or curve composite entry. Section index :  " << loop.domain_curve << ".\n";
          }
        }
      }
      else {
        BOOST_LOG_TRIVIAL(error) << "igs_loader::_apply_trim_loop(): Unsupported loop type. Type :  " << loop.preferred_trim_type << ".\n";
      }
    }
    else {
      BOOST_LOG_TRIVIAL(error) << "igs_loader::_apply_trim_loop(): Missing curve on surface. Section index :  " << section_index << ".\n";
    }
  }

  ////////////////////////////////////////////////////////////////////////////////
  math::nurbscurve2d igs_loader::_migrate_curve(math::nurbscurve3d const& nc3d)
  {
    math::nurbscurve2d nc2d;

    // apply same degree and knotvector
    nc2d.degree(nc3d.degree());
    nc2d.set_knotvector(nc3d.knots().begin(), nc3d.knots().end());

    // copy 3d -> 2d points to nurbscurve control polygon
    std::vector<math::point2d> points2d;
    for (auto const& p3d : nc3d.points()) {
      points2d.push_back(math::point2d (p3d[math::point2d::x], p3d[math::point3d::y], p3d.weight()));
    }
    
    nc2d.set_points(points2d.begin(), points2d.end());
    return nc2d;
  }

} // namespace gpucast

