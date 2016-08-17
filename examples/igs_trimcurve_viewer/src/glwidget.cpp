/********************************************************************************
*
* Copyright (C) 2010 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : glwidget.cpp
*  project    : glpp
*  description:
*
********************************************************************************/
#include "glwidget.hpp"

#pragma warning(disable: 4127) // Qt conditional expression is constant

#include <mainwindow.hpp>

#include <QtGui/QKeyEvent>

// system includes
//#include <QtGui/QMouseEvent>
//#include <QtOpenGL/QGLFormat>
//
//#include <CL/cl.hpp>
//
#include <sstream>
#include <iostream>
#include <cctype>
#include <typeinfo>

#include <gpucast/gl/util/init_glew.hpp>
#include <gpucast/gl/util/contextinfo.hpp>
#include <gpucast/gl/util/timer.hpp>

#include <gpucast/math/vec2.hpp>
#include <gpucast/math/vec4.hpp>
#include <gpucast/math/util/prefilter2d.hpp>

#include <gpucast/core/config.hpp>

#include <gpucast/gl/shader.hpp>
#include <gpucast/gl/elementarraybuffer.hpp>
#include <gpucast/gl/arraybuffer.hpp>
#include <gpucast/gl/error.hpp>
#include <gpucast/gl/util/transferfunction.hpp>
#include <gpucast/gl/util/resource_factory.hpp>

#include <boost/filesystem.hpp>
#include <boost/unordered_map.hpp>

#include <gpucast/core/surface_converter.hpp>
#include <gpucast/core/import/igs.hpp>
#include <gpucast/core/import/igs_loader.hpp>
#include <gpucast/core/trimdomain.hpp>
#include <gpucast/core/trimdomain_serializer_double_binary.hpp>
#include <gpucast/core/trimdomain_serializer_contour_map_binary.hpp>
#include <gpucast/core/trimdomain_serializer_contour_map_kd.hpp>
#include <gpucast/core/trimdomain_serializer_loop_contour_list.hpp>

#include <gpucast/math/parametric/domain/partition/monotonic_contour/contour_map_binary.hpp>
#include <gpucast/math/parametric/domain/partition/monotonic_contour/contour_map_loop_list.hpp>
#include <gpucast/math/parametric/domain/partition/monotonic_contour/contour_map_kd.hpp>
#include <gpucast/math/parametric/domain/partition/double_binary/partition.hpp>

using namespace std;

///////////////////////////////////////////////////////////////////////
glwidget::glwidget( int argc, char** argv, QGLFormat const& context_format, QWidget *parent)
 :  QGLWidget                 ( context_format, parent),
    _argc                     ( argc ),
    _argv                     ( argv ),
    _width                    ( 1 ),
    _height                   ( 1 ),
    _initialized              ( false ),
    _partition_program        ( nullptr ),
    _db_program               ( nullptr ),
    _db_trimdata              ( nullptr ),
    _db_celldata              ( nullptr ),
    _db_curvelist             ( nullptr ),
    _db_curvedata             ( nullptr ),
    _cmb_program              ( nullptr ),
    _cmb_partition            ( nullptr ),
    _cmb_contourlist          ( nullptr ),
    _cmb_curvelist            ( nullptr ),
    _cmb_curvedata            ( nullptr ),
    _cmb_pointdata            ( nullptr ),
    _kd_program               ( nullptr ),
    _kd_partition             ( nullptr ),
    _kd_contourlist           ( nullptr ),
    _kd_curvelist             ( nullptr ),
    _kd_curvedata             ( nullptr ),
    _kd_pointdata             ( nullptr ),
    _loop_list_loops          ( nullptr ),
    _loop_list_contours       ( nullptr ),
    _loop_list_curves         ( nullptr ),
    _loop_list_points         ( nullptr ),
    _quad                     ( nullptr ),
    _transfertexture          ( nullptr ),
    _curve_geometry           ( ),
    _view                     ( original ),
    _show_texel_fetches       ( false )
{
  setFocusPolicy(Qt::StrongFocus);
}


///////////////////////////////////////////////////////////////////////
glwidget::~glwidget()
{}

///////////////////////////////////////////////////////////////////////
void                    
glwidget::open ( std::list<std::string> const& files )
{
  for ( std::string const& filename : files )
  {
    gpucast::igs_loader igs_loader; 
    auto objects = igs_loader.load(filename);
    
    // merge bezier objects
    std::shared_ptr<gpucast::beziersurfaceobject> bezier_object = std::make_shared<gpucast::beziersurfaceobject>();
    _objects[filename] = bezier_object;

    for (auto nurbs_object : objects) {
      std::shared_ptr<gpucast::beziersurfaceobject> tmp = std::make_shared<gpucast::beziersurfaceobject>();

      gpucast::surface_converter converter;
      converter.convert(nurbs_object, tmp);
      bezier_object->merge(*tmp);
    }
  }
}


///////////////////////////////////////////////////////////////////////
void                    
glwidget::clear() 
{
  _curve_geometry.clear();
}


///////////////////////////////////////////////////////////////////////
void                    
glwidget::update_view(std::string const& name, 
                      std::size_t const index, 
                      view current_view, 
                      unsigned resolution)
{
  _current_object        = name;
  _current_surface       = index;
  _view                  = current_view;
  _current_texresolution = resolution;

  clear();

  initialize_sampler();
  _initialize_prefilter();

  // create new polygons
  if ( _objects.count( name ) )
  {
    std::size_t nsurfaces = _objects.find(name)->second->size();
    if ( nsurfaces > index )
    {
      auto surface = _objects.find(name)->second->begin();
      std::advance ( surface, index );
    
      // set projection
      auto domain = (**surface).domain();

      _domain_size = gpucast::math::vec2f(domain->nurbsdomain().size());
      _domain_min = gpucast::math::vec2f(domain->nurbsdomain().min);

      mainwindow* win = dynamic_cast<mainwindow*>(parent());
      if (!win) {
        throw std::runtime_error("No Main Window");
      }
      win->show_domainsize(_domain_min[0], _domain_min[1], _domain_min[0] + _domain_size[0], _domain_min[1] + _domain_size[1]);

      std::size_t bytes = 0;

      switch ( _view )
      {
        case original :
          generate_original_view ( domain );
          break;
        case double_binary_partition :
          generate_double_binary_view ( domain );
          break;
        case double_binary_classification :
          bytes = serialize_double_binary(domain);
          break;
        case contour_map_binary_partition :
          generate_bboxmap_view ( domain );
          break;
        case contour_map_binary_classification :
          bytes = serialize_contour_binary(domain);
          break;
        case contour_map_kd_classification:
          bytes = serialize_contour_kd(domain, win->kdsplit());
          break;
        case contour_map_kd_partition:
          generate_kd_view(domain, win->kdsplit());
          break;
        case minification:
          generate_minification_view(domain, resolution);
          break;
        case contour_map_loop_list_partition:
          generate_loop_list_view(domain);
          break;
        case contour_map_loop_list_classification:
          bytes = serialize_contour_loop_list(domain);
          break;
        case distance_field:
          generate_distance_field(domain, resolution);
          break;
        case binary_field:
          generate_binary_field(domain, resolution);
          break;
        case prefilter:
          
          break;
        default : 
          break;
      };

      for (auto const& r : _testruns) {
        if (r->rendermode == current_view) {
          r->data[r->current].size_bytes = bytes;
        }
      }

      _projection = gpucast::math::ortho(float(domain->nurbsdomain().min[gpucast::trimdomain::point_type::u] ), 
                                         float(domain->nurbsdomain().max[gpucast::trimdomain::point_type::u] ), 
                                         float(domain->nurbsdomain().min[gpucast::trimdomain::point_type::v] ), 
                                         float(domain->nurbsdomain().max[gpucast::trimdomain::point_type::v] ), 
                                         0.1f, 
                                         10.0f);
    }
  }
}


///////////////////////////////////////////////////////////////////////
void                    
glwidget::generate_original_view ( gpucast::beziersurface::trimdomain_ptr const& domain )
{
  gpucast::trimdomain::curve_container curves = domain->curves();

  for ( auto curve = curves.begin(); curve != curves.end(); ++curve )
  {
    gpucast::math::vec4f cpolygon_color ( 1.0f, 1.0f, 1.0f, 1.0f );
    add_gl_curve ( **curve, cpolygon_color );

    // generate bbox to draw
    gpucast::math::bbox2d bbox;
    (**curve).bbox_simple(bbox);

    gpucast::math::vec4f bbox_color ( 1.0f, 0.0f, 0.0f, 1.0f );
    add_gl_bbox ( bbox, bbox_color );
  }
}


///////////////////////////////////////////////////////////////////////
void                    
glwidget::generate_double_binary_view ( gpucast::beziersurface::trimdomain_ptr const& domain )
{
  generate_trim_region_vbo(domain);

  auto curves = domain->curves();
  gpucast::math::domain::partition<gpucast::beziersurface::curve_point_type>  partition(curves.begin(), curves.end());
  partition.initialize();

  for ( auto v = partition.begin(); v != partition.end(); ++v )
  {
    gpucast::math::bbox2d bbox ( gpucast::math::point2d ( (**v).get_horizontal_interval().minimum(), (**v).get_vertical_interval().minimum() ),
                                 gpucast::math::point2d ( (**v).get_horizontal_interval().maximum(), (**v).get_vertical_interval().maximum() ) );
      gpucast::math::vec4f cell_color ( 1.0f, 1.0f, 0.0f, 1.0f );
      add_gl_bbox ( bbox, cell_color ); 
    for ( auto c = (**v).begin(); c != (**v).end(); ++c )
    {
      gpucast::math::bbox2d bbox ( gpucast::math::point2d ( (**c).get_horizontal_interval().minimum(), (**c).get_vertical_interval().minimum() ),
                                   gpucast::math::point2d ( (**c).get_horizontal_interval().maximum(), (**c).get_vertical_interval().maximum() ) );
      gpucast::math::vec4f cell_color ( 0.0f, 1.0f, 0.0f, 1.0f );
      add_gl_bbox ( bbox, cell_color ); 
    }
  }
}


///////////////////////////////////////////////////////////////////////
void                    
glwidget::generate_bboxmap_view ( gpucast::beziersurface::trimdomain_ptr const& domain )
{
  gpucast::math::domain::contour_map_binary<double> cmap;
  for ( auto const& loop : domain->loops() )
  {
    cmap.add ( gpucast::math::domain::contour<double> ( loop.begin(), loop.end() ) );
  }

  cmap.initialize();

  for ( gpucast::math::domain::contour_map_binary<double>::contour_segment_ptr const& segment : cmap.monotonic_segments() )
  {
    gpucast::math::vec4f segment_color ( 0.0f, 1.0f, 1.0f, 1.0f );
    add_gl_bbox ( segment->bbox(), segment_color );

    for ( auto c = segment->begin(); c != segment->end(); ++c ) 
    {
      gpucast::math::vec4f curve_bbox_color ( 1.0f, 0.0f, 0.0f, 1.0f );
      gpucast::math::bbox2d bbox;
      (**c).bbox_simple(bbox);
      add_gl_bbox ( bbox, curve_bbox_color );

      gpucast::math::vec4f cpolygon_color ( 1.0f, 1.0f, 1.0f, 1.0f );
      add_gl_curve ( **c, cpolygon_color );
    }
  }

  for ( gpucast::math::domain::contour_map_binary<double>::contour_interval const& vslab : cmap.partition() )
  {
    for ( gpucast::math::domain::contour_map_binary<double>::contour_cell const& cell : vslab.cells )
    {
      gpucast::math::vec4f vslab_color ( 0.0f, 1.0f, 0.0f, 1.0f );
      gpucast::math::bbox2d vbox ( gpucast::math::point2d ( cell.interval_u.minimum(), cell.interval_v.minimum()), gpucast::math::point2d ( cell.interval_u.maximum(), cell.interval_v.maximum()));
       add_gl_bbox ( vbox, vslab_color );
      for ( gpucast::math::domain::contour_map_binary<double>::contour_segment_ptr const& contour : cell.overlapping_segments )
      {
        gpucast::math::vec4f cell_color ( 0.0f, 1.0f, 0.0f, 1.0f );
        gpucast::math::bbox2d ubox = contour->bbox();
        add_gl_bbox ( ubox, cell_color );
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////
void
glwidget::generate_loop_list_view(gpucast::beziersurface::trimdomain_ptr const& domain)
{
  generate_trim_region_vbo(domain);

  gpucast::math::domain::contour_map_loop_list<double> looplist;

  for (auto const& loop : domain->loops()) {
    looplist.add(gpucast::math::domain::contour<double>(loop.begin(), loop.end()));
  }

  bool success = looplist.initialize();

  for (auto const& contour : looplist.loops()) {
    gpucast::math::vec4f loop_color(1.0f, 0.0f, 0.0f, 1.0f);
    gpucast::math::bbox2d ubox = contour.first->bbox();
    add_gl_bbox(ubox, loop_color);

    for (auto const& segment : contour.second) {
      gpucast::math::vec4f segment_color(0.0f, 1.0f, 0.0f, 1.0f);
      gpucast::math::bbox2d ubox = segment->bbox();
      add_gl_bbox(ubox, segment_color);
    }
  }
}

///////////////////////////////////////////////////////////////////////
void
glwidget::generate_kd_view(gpucast::beziersurface::trimdomain_ptr const& domain, gpucast::kd_split_strategy s)
{
  using namespace gpucast::math::domain;
  generate_trim_region_vbo(domain);

  contour_map_kd<double> kdmap(s);

  for (auto const& loop : domain->loops()) {
    kdmap.add(contour<double>(loop.begin(), loop.end()));
  }
  if (!kdmap.initialize()) {
    std::cout << "initialization failed!\n";
  }

  std::vector<std::shared_ptr<kdnode2d<double>>> nodes;

  if (!kdmap.partition().root) {
    return;
  }

  kdmap.partition().root->serialize_bfs(nodes);

  unsigned tmp_count = 0;

  std::map<unsigned, gpucast::math::vec4f> cmap;
  cmap[0] = gpucast::math::vec4f{ 1.0f, 0.0f, 0.0f, 1.0f };
  cmap[1] = gpucast::math::vec4f{ 1.0f, 1.0f, 0.0f, 1.0f };
  cmap[2] = gpucast::math::vec4f{ 1.0f, 1.0f, 1.0f, 1.0f };
  cmap[3] = gpucast::math::vec4f{ 1.0f, 0.0f, 1.0f, 1.0f };
  cmap[4] = gpucast::math::vec4f{ 0.0f, 1.0f, 1.0f, 1.0f };
  cmap[5] = gpucast::math::vec4f{ 0.0f, 1.0f, 0.0f, 1.0f };
  cmap[6] = gpucast::math::vec4f{ 0.0f, 0.0f, 1.0f, 1.0f };
  cmap[7] = gpucast::math::vec4f{ 0.0f, 0.0f, 0.5f, 1.0f };
  cmap[8] = gpucast::math::vec4f{ 0.5f, 0.0f, 0.0f, 1.0f };
  cmap[9] = gpucast::math::vec4f{ 0.0f, 0.5f, 0.0f, 1.0f };
  cmap[10] = gpucast::math::vec4f{ 0.5f, 0.5f, 0.0f, 1.0f };
  cmap[11] = gpucast::math::vec4f{ 0.5f, 0.5f, 0.5f, 1.0f };
  cmap[12] = gpucast::math::vec4f{ 0.0f, 0.5f, 0.5f, 1.0f };

  for (auto const& kdnode : nodes) 
  {
    if (kdnode->is_leaf()) {
      unsigned cindex = tmp_count % cmap.size();
      auto col = cmap[cindex];

      gpucast::math::bbox2d ubox = kdnode->bbox;
      add_gl_bbox(ubox, col, true);
      /*
      std::cout << "leaf : " << std::endl;
      std::cout << "   - display color : " << col << std::endl;
      std::cout << "   - # : " << tmp_count++ << std::endl;
      std::cout << "   - segments : " << kdnode->overlapping_segments.size() << std::endl;
      std::cout << "   - parity : " << kdnode->parity << std::endl;
      std::cout << "   - bbox : " << kdnode->bbox << std::endl;
      std::cout << "   - size : " << kdnode->bbox.size() << std::endl;*/
    }
    if (!kdnode->is_leaf())
    {
      if (kdnode->split_direction == gpucast::math::point2d::u) {
        gpucast::math::beziercurve2d bc;
        bc.add(gpucast::math::point2d(kdnode->split_value, kdnode->bbox.min[gpucast::math::point2d::v]));
        bc.add(gpucast::math::point2d(kdnode->split_value, kdnode->bbox.max[gpucast::math::point2d::v]));
        add_gl_curve(bc, gpucast::math::vec4f{ 1.0f, 0.0f, 0.0f, 1.0f });
      }
      else {
        gpucast::math::beziercurve2d bc;
        bc.add(gpucast::math::point2d(kdnode->bbox.min[gpucast::math::point2d::u], kdnode->split_value));
        bc.add(gpucast::math::point2d(kdnode->bbox.max[gpucast::math::point2d::u], kdnode->split_value));
        add_gl_curve(bc, gpucast::math::vec4f{ 0.0f, 0.0f, 1.0f, 1.0f });
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////
void
glwidget::generate_minification_view(gpucast::beziersurface::trimdomain_ptr const& domain, unsigned resolution)
{
  generate_trim_region_vbo(domain);

  unsigned res_u = resolution;
  unsigned res_v = resolution;

  auto distance_field = domain->signed_distance_field(resolution);

  auto offset_u = (domain->nurbsdomain().max[gpucast::trimdomain::point_type::u] - domain->nurbsdomain().min[gpucast::trimdomain::point_type::u]) / res_u;
  auto offset_v = (domain->nurbsdomain().max[gpucast::trimdomain::point_type::v] - domain->nurbsdomain().min[gpucast::trimdomain::point_type::v]) / res_v;

  auto min_u = domain->nurbsdomain().min[gpucast::trimdomain::point_type::u];
  auto min_v = domain->nurbsdomain().min[gpucast::trimdomain::point_type::v];
  
  // draw sampled regions
  for (unsigned u = 0; u < res_u; ++u) 
  {
    for (unsigned v = 0; v < res_v; ++v) 
    {
      auto bbox_min = gpucast::trimdomain::point_type(min_u + u*offset_u ,     min_v + v*offset_v);
      auto bbox_max = gpucast::trimdomain::point_type(min_u + (u+1)*offset_u , min_v + (v+1)*offset_v);

      gpucast::math::bbox2d ubox(bbox_min, bbox_max);
      auto minimal_distance = distance_field[v*resolution + u];

      float intensity = std::fabs(0.1f + minimal_distance / std::max(offset_u, offset_v));
      gpucast::math::vec4f color_inside(0.0f, intensity, 0.0f, 1.0f);
      gpucast::math::vec4f color_outside(intensity, 0.0f, 0.0f, 1.0f);

      if (minimal_distance > 0) {
        add_gl_bbox(ubox, color_inside);
      }
      else {
        add_gl_bbox(ubox, color_outside);
      }
    }
  }
}


///////////////////////////////////////////////////////////////////////
void
glwidget::generate_binary_field(gpucast::beziersurface::trimdomain_ptr const& domain, unsigned resolution)
{
  generate_trim_region_vbo(domain);

  initialize_sampler();

  _binary_texture = make_unique<gpucast::gl::texture2d>();
  std::vector<float> texture_data;

  unsigned res_u = resolution;
  unsigned res_v = resolution;

  auto offset_u = (domain->nurbsdomain().max[gpucast::trimdomain::point_type::u] - domain->nurbsdomain().min[gpucast::trimdomain::point_type::u]) / res_u;
  auto offset_v = (domain->nurbsdomain().max[gpucast::trimdomain::point_type::v] - domain->nurbsdomain().min[gpucast::trimdomain::point_type::v]) / res_v;

  auto min_u = domain->nurbsdomain().min[gpucast::trimdomain::point_type::u];
  auto min_v = domain->nurbsdomain().min[gpucast::trimdomain::point_type::v];

  // draw sampled regions
  for (unsigned v = 0; v < res_v; ++v)
  {
    for (unsigned u = 0; u < res_u; ++u)
    {
      auto bbox_min = gpucast::trimdomain::point_type(min_u + u*offset_u, min_v + v*offset_v);
      auto bbox_max = gpucast::trimdomain::point_type(min_u + (u + 1)*offset_u, min_v + (v + 1)*offset_v);

      gpucast::math::bbox2d ubox(bbox_min, bbox_max);

      unsigned intersections = 0;
      for (auto const& loop : domain->loops()) {
        intersections += loop.is_inside(ubox.center());
      }

      if (intersections % 2) {
        texture_data.push_back(1.0f);
      }
      else {
        texture_data.push_back(-1.0f);
      }
    }
  }

  _binary_texture->teximage(0, GL_R32F, res_u, res_v, 0, GL_RED, GL_FLOAT, (void*)(&texture_data[0]));
}

///////////////////////////////////////////////////////////////////////
void
glwidget::generate_distance_field(gpucast::beziersurface::trimdomain_ptr const& domain, unsigned resolution)
{
  generate_trim_region_vbo(domain);

  initialize_sampler();

  _distance_field_texture = make_unique<gpucast::gl::texture2d>();
   
  auto distance_field = domain->signed_distance_field(resolution);
  std::vector<float> distance_fieldf(distance_field.begin(), distance_field.end());

  _distance_field_texture->teximage(0, GL_R32F, resolution, resolution, 0, GL_RED, GL_FLOAT, (void*)(&distance_fieldf[0]));
}


///////////////////////////////////////////////////////////////////////
std::size_t                    
glwidget::serialize_double_binary ( gpucast::beziersurface::trimdomain_ptr const& domain )
{
  if ( !_db_trimdata )    _db_trimdata     = make_unique<gpucast::gl::texturebuffer>();
  if ( !_db_celldata )    _db_celldata     = make_unique<gpucast::gl::texturebuffer>();
  if ( !_db_curvelist )   _db_curvelist    = make_unique<gpucast::gl::texturebuffer>();
  if ( !_db_curvedata )   _db_curvedata    = make_unique<gpucast::gl::texturebuffer>();
  if ( !_db_preclassdata) _db_preclassdata = make_unique<gpucast::gl::texturebuffer>();

  gpucast::trimdomain_serializer_double_binary serializer;
  gpucast::trim_double_binary_serialization serialization;
  _trimid = serializer.serialize(domain, serialization, _tex_classification_enabled, _current_texresolution);

  _db_trimdata->update(serialization.partition.begin(), serialization.partition.end());
  _db_celldata->update(serialization.celldata.begin(), serialization.celldata.end());
  _db_curvelist->update(serialization.curvelist.begin(), serialization.curvelist.end());
  _db_curvedata->update(serialization.curvedata.begin(), serialization.curvedata.end());
  _db_preclassdata->update(serialization.preclassification.begin(), serialization.preclassification.end());
                       
  _db_trimdata->format  ( GL_RGBA32F );
  _db_celldata->format  ( GL_RGBA32F );
  _db_curvelist->format ( GL_RGBA32F );
  _db_curvedata->format ( GL_RGB32F );
  _db_preclassdata->format (GL_R8UI );

  // overdraw outer box
  gpucast::math::vec4f loop_color(1.0f, 1.0f, 0.0f, 1.0f);
  auto box = domain->nurbsdomain();
  add_gl_bbox(box, loop_color);

  // show memory usage
  std::size_t size_bytes = serialization.size_in_bytes();

  mainwindow* win = dynamic_cast<mainwindow*>(parent());
  if (win) {
    win->show_memusage(size_bytes);
  }

  return size_bytes;
}

///////////////////////////////////////////////////////////////////////
std::size_t
glwidget::serialize_contour_kd(gpucast::beziersurface::trimdomain_ptr const& domain, gpucast::kd_split_strategy split_strategy)
{
  if (!_kd_partition)   _kd_partition = make_unique<gpucast::gl::texturebuffer>();
  if (!_kd_contourlist) _kd_contourlist = make_unique<gpucast::gl::texturebuffer>();
  if (!_kd_curvelist) _kd_curvelist = make_unique<gpucast::gl::texturebuffer>();
  if (!_kd_curvedata) _kd_curvedata = make_unique<gpucast::gl::texturebuffer>();
  if (!_kd_pointdata) _kd_pointdata = make_unique<gpucast::gl::texturebuffer>();
  if (!_kd_preclassdata) _kd_preclassdata = make_unique<gpucast::gl::texturebuffer>();

  gpucast::trimdomain_serializer_contour_map_kd serializer;
  gpucast::trim_kd_serialization serialization;

  _trimid = serializer.serialize(domain,
    split_strategy,
    serialization,
    _tex_classification_enabled, 
    _current_texresolution);

  _kd_partition->update(serialization.partition.begin(), serialization.partition.end());
  _kd_contourlist->update(serialization.contourlist.begin(), serialization.contourlist.end());
  _kd_curvelist->update(serialization.curvelist.begin(), serialization.curvelist.end());
  _kd_curvedata->update(serialization.curvedata.begin(), serialization.curvedata.end());
  _kd_pointdata->update(serialization.pointdata.begin(), serialization.pointdata.end());
  _kd_preclassdata->update(serialization.preclassification.begin(), serialization.preclassification.end());

  _kd_partition->format(GL_RGBA32F);
  _kd_contourlist->format(GL_RGBA32F);
  _kd_curvelist->format(GL_RGBA32F);
  _kd_curvedata->format(GL_R32F);
  _kd_pointdata->format(GL_RGB32F);
  _kd_preclassdata->format(GL_R8UI);

  // overdraw outer box
  gpucast::math::vec4f loop_color(1.0f, 1.0f, 0.0f, 1.0f);
  auto box = domain->nurbsdomain();
  add_gl_bbox(box, loop_color);

  // show memory usage
  std::size_t size_bytes = serialization.size_in_bytes();

  mainwindow* win = dynamic_cast<mainwindow*>(parent());
  if (win) {
    win->show_memusage(size_bytes);
  }

  return size_bytes;
}

///////////////////////////////////////////////////////////////////////
std::size_t
glwidget::serialize_contour_binary ( gpucast::beziersurface::trimdomain_ptr const& domain )
{
  if (!_cmb_partition)   _cmb_partition = make_unique<gpucast::gl::texturebuffer>();
  if (!_cmb_contourlist) _cmb_contourlist = make_unique<gpucast::gl::texturebuffer>();
  if (!_cmb_curvelist) _cmb_curvelist = make_unique<gpucast::gl::texturebuffer>();
  if (!_cmb_curvedata) _cmb_curvedata = make_unique<gpucast::gl::texturebuffer>();
  if (!_cmb_pointdata) _cmb_pointdata = make_unique<gpucast::gl::texturebuffer>();
  if (!_cmb_preclassdata) _cmb_preclassdata = make_unique<gpucast::gl::texturebuffer>();

  gpucast::trimdomain_serializer_contour_map_binary serializer;
  gpucast::trim_contour_binary_serialization serialization;
  _trimid = serializer.serialize ( domain, 
                                   serialization,
                                   _tex_classification_enabled,
                                   _current_texresolution);

  _cmb_partition->update   ( serialization.partition.begin(),   serialization.partition.end());
  _cmb_contourlist->update ( serialization.contourlist.begin(), serialization.contourlist.end());
  _cmb_curvelist->update   ( serialization.curvelist.begin(),   serialization.curvelist.end());
  _cmb_curvedata->update   ( serialization.curvedata.begin(),   serialization.curvedata.end());
  _cmb_pointdata->update   ( serialization.pointdata.begin(),   serialization.pointdata.end());
  _cmb_preclassdata->update(serialization.preclassification.begin(), serialization.preclassification.end());

  _cmb_partition->format   ( GL_RGBA32F );
  _cmb_contourlist->format ( GL_RGBA32F );
  _cmb_curvelist->format   ( GL_RGBA32F );
  _cmb_curvedata->format   ( GL_R32F );
  _cmb_pointdata->format   ( GL_RGB32F );
  _cmb_preclassdata->format( GL_R8UI );

  // overdraw outer box
  gpucast::math::vec4f loop_color(1.0f, 1.0f, 0.0f, 1.0f);
  auto box = domain->nurbsdomain();
  add_gl_bbox(box, loop_color);

  // show memory usage
  std::size_t size_bytes = serialization.size_in_bytes();

  mainwindow* win = dynamic_cast<mainwindow*>(parent());
  if (win) {
    win->show_memusage(size_bytes);
  }

  return size_bytes;
}


///////////////////////////////////////////////////////////////////////
std::size_t
glwidget::serialize_contour_loop_list(gpucast::beziersurface::trimdomain_ptr const& domain)
{
  if (!_loop_list_loops)    _loop_list_loops = make_unique<gpucast::gl::shaderstoragebuffer>();
  if (!_loop_list_contours) _loop_list_contours = make_unique<gpucast::gl::shaderstoragebuffer>();
  if (!_loop_list_curves)   _loop_list_curves = make_unique<gpucast::gl::shaderstoragebuffer>();
  if (!_loop_list_points)   _loop_list_points = make_unique<gpucast::gl::shaderstoragebuffer>();
  if (!_loop_list_preclassdata) _loop_list_preclassdata = make_unique<gpucast::gl::texturebuffer>();

  gpucast::trim_loop_list_serialization serialization;
  gpucast::trimdomain_serializer_loop_contour_list serializer;

  std::unordered_map<gpucast::beziersurface::trimdomain_ptr, gpucast::trimdomain_serializer::address_type> referenced_domains;

  _trimid = serializer.serialize(domain, serialization, _tex_classification_enabled, _current_texresolution);

  // write data to shader storage
  _loop_list_loops->update(serialization.loops.begin(), serialization.loops.end());
  _loop_list_contours->update(serialization.contours.begin(), serialization.contours.end());
  _loop_list_curves->update(serialization.curves.begin(), serialization.curves.end());
  _loop_list_points->update(serialization.points.begin(), serialization.points.end());
  _loop_list_preclassdata->update(serialization.preclassification.begin(), serialization.preclassification.end());

  _loop_list_preclassdata->format(GL_R8UI);

  // overdraw outer box
  gpucast::math::vec4f loop_color(1.0f, 1.0f, 0.0f, 1.0f);
  auto box = domain->nurbsdomain();
  add_gl_bbox(box, loop_color);

  // show memory usage
  std::size_t size_bytes = serialization.size_in_bytes();

  mainwindow* win = dynamic_cast<mainwindow*>(parent());
  if (win) {
    win->show_memusage(size_bytes);
  }

  return size_bytes;
}

///////////////////////////////////////////////////////////////////////
void glwidget::initialize_sampler()
{
  if (!_bilinear_filter) {
    _bilinear_filter = make_unique<gpucast::gl::sampler>();
    _bilinear_filter->parameter(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    _bilinear_filter->parameter(GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    //_bilinear_filter->parameter(GL_TEXTURE_WRAP_S, GL_CLAMP);
    //_bilinear_filter->parameter(GL_TEXTURE_WRAP_T, GL_CLAMP);
    //_bilinear_filter->parameter(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    //_bilinear_filter->parameter(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    _bilinear_filter->parameter(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    _bilinear_filter->parameter(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    //_bilinear_filter->parameter(GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
    //_bilinear_filter->parameter(GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
  } 

  if (!_nearest_filter) {
    _nearest_filter = make_unique<gpucast::gl::sampler>();
    _nearest_filter->parameter(GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    _nearest_filter->parameter(GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    //_nearest_filter->parameter(GL_TEXTURE_WRAP_S, GL_CLAMP);
    //_nearest_filter->parameter(GL_TEXTURE_WRAP_T, GL_CLAMP);
    //_nearest_filter->parameter(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    //_nearest_filter->parameter(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    _nearest_filter->parameter(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    _nearest_filter->parameter(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    //_nearest_filter->parameter(GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
    //_nearest_filter->parameter(GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
  }
}

///////////////////////////////////////////////////////////////////////
void glwidget::generate_trim_region_vbo(gpucast::beziersurface::trimdomain_ptr const& domain)
{
  gpucast::trimdomain::curve_container curves = domain->curves();
  for (auto curve = curves.begin(); curve != curves.end(); ++curve)
  {
    gpucast::math::vec4f cpolygon_color(1.0f, 1.0f, 0.0f, 1.0f);
    add_gl_curve(**curve, cpolygon_color);
  }
}


///////////////////////////////////////////////////////////////////////
void                    
glwidget::add_gl_curve ( gpucast::beziersurface::curve_type const& curve, gpucast::math::vec4f const& color )
{
  // draw curve using a fixed number of sample points
  unsigned const samples = 50;
  std::vector<gpucast::math::vec4f> curve_points;
  for ( unsigned sample = 0; sample <= samples; ++sample ) 
  {
    gpucast::math::point2d p;
    curve.evaluate(float(sample)/samples, p);
    curve_points.push_back( gpucast::math::vec4f ( p[0], p[1], -1.0f, 1.0f ) );
  }

  std::vector<gpucast::math::vec4f> curve_colors ( curve_points.size(), color );
  auto gl_curve = std::make_shared<gpucast::gl::line> ( curve_points, 0, 1, 2);
  gl_curve->set_color ( curve_colors );
  _curve_geometry.push_back ( gl_curve ); 

}


///////////////////////////////////////////////////////////////////////
void                    
glwidget::add_gl_bbox(gpucast::math::bbox2d const& bbox, gpucast::math::vec4f const& color, bool diagonals)
{
  std::vector<gpucast::math::vec4f> bbox_points;

  // outer boundary
  bbox_points.push_back(gpucast::math::vec4f(float(bbox.min[0]), float(bbox.min[1]), -2.0f, 1.0f));
  bbox_points.push_back(gpucast::math::vec4f(float(bbox.max[0]), float(bbox.min[1]), -2.0f, 1.0f));
  bbox_points.push_back(gpucast::math::vec4f(float(bbox.max[0]), float(bbox.min[1]), -2.0f, 1.0f));
  bbox_points.push_back(gpucast::math::vec4f(float(bbox.max[0]), float(bbox.max[1]), -2.0f, 1.0f));
  bbox_points.push_back(gpucast::math::vec4f(float(bbox.max[0]), float(bbox.max[1]), -2.0f, 1.0f));
  bbox_points.push_back(gpucast::math::vec4f(float(bbox.min[0]), float(bbox.max[1]), -2.0f, 1.0f));
  bbox_points.push_back(gpucast::math::vec4f(float(bbox.min[0]), float(bbox.max[1]), -2.0f, 1.0f));
  bbox_points.push_back(gpucast::math::vec4f(float(bbox.min[0]), float(bbox.min[1]), -2.0f, 1.0f));

  // diagonals
  if (diagonals) {
    bbox_points.push_back(gpucast::math::vec4f(float(bbox.min[0]), float(bbox.min[1]), -2.0f, 1.0f));
    bbox_points.push_back(gpucast::math::vec4f(float(bbox.max[0]), float(bbox.max[1]), -2.0f, 1.0f));
    bbox_points.push_back(gpucast::math::vec4f(float(bbox.min[0]), float(bbox.max[1]), -2.0f, 1.0f));
    bbox_points.push_back(gpucast::math::vec4f(float(bbox.max[0]), float(bbox.min[1]), -2.0f, 1.0f));
  }
  std::vector<gpucast::math::vec4f> bbox_colors ( bbox_points.size(), color );

  auto gl_bbox = std::make_shared<gpucast::gl::line>(bbox_points, 0, 1, 2);
  gl_bbox->set_color ( bbox_colors );
  _curve_geometry.push_back ( gl_bbox );
}


///////////////////////////////////////////////////////////////////////
glwidget::trimdomain_ptr
glwidget::get_domain ( std::string const& name, std::size_t const index ) const
{
  if ( _objects.count( name ) )
  {
    if ( _objects.find(name)->second->size() > index )
    {
      auto surface = _objects.find(name)->second->begin();
      std::advance ( surface, index );
      return (**surface).domain();
    } else {
      std::cerr << "Irregular surface index\n";
    } 
  } else {
    std::cerr << "Irregular object name\n";
  }
  return trimdomain_ptr();
}


///////////////////////////////////////////////////////////////////////
std::size_t                    
glwidget::get_objects () const 
{
  return _objects.size();
}                    


///////////////////////////////////////////////////////////////////////
std::size_t 
glwidget::get_surfaces ( std::string const& name ) const
{
  if ( _objects.count(name) )
  {
    return _objects.find(name)->second->surfaces();
  } else {
    return 0;
  }
}

///////////////////////////////////////////////////////////////////////
void                    
glwidget::pixel_size(unsigned s)
{
  _pixel_size = s;
}

///////////////////////////////////////////////////////////////////////
void
glwidget::show_texel_fetches(int enable)
{
  _show_texel_fetches = enable != 0;
}

///////////////////////////////////////////////////////////////////////
void                    
glwidget::tex_classification(int enable)
{
  _tex_classification_enabled = enable;

  if(dynamic_cast<mainwindow*>(parent())) {
    dynamic_cast<mainwindow*>(parent())->update_view();
  }
}

///////////////////////////////////////////////////////////////////////
void
glwidget::optimal_distance(int enable)
{
  _optimal_distance = enable != 0;

  if (dynamic_cast<mainwindow*>(parent())) {
    dynamic_cast<mainwindow*>(parent())->update_view();
  }
}

///////////////////////////////////////////////////////////////////////
void
glwidget::antialiasing(enum aamode mode)
{
  _aamode = mode;
}

///////////////////////////////////////////////////////////////////////
void
glwidget::rendermode(enum view mode)
{
  _view = mode;
}

///////////////////////////////////////////////////////////////////////
void                    
glwidget::zoom(float scale) {
  _zoom = scale;
}

///////////////////////////////////////////////////////////////////////
void                    
glwidget::recompile()
{
  std::cout << "Recompiling shader code." << std::endl;
  _initialize_shader();
}

///////////////////////////////////////////////////////////////////////
void
glwidget::resetview()
{
  _shift_x = 0.0f;
  _shift_y = 0.0f;
  _zoom = 1.0f;
}

///////////////////////////////////////////////////////////////////////
void glwidget::testrun(std::list<std::string> const& objects)
{
  _initialize_prefilter();
#if 0
  testrun_t run_db = { double_binary_classification, _aamode, 0U, {}, 0.0 };
  testrun_t run_cb = { contour_map_binary_classification, _aamode, 0U, {}, 0.0 };
  testrun_t run_ll = { contour_map_loop_list_classification, _aamode, 0U, {}, 0.0 };
  testrun_t run_kd = { contour_map_kd_classification, _aamode, 0U, {}, 0.0 };

  _testruns.push_back(std::make_shared<testrun_t>(run_db));
  _testruns.push_back(std::make_shared<testrun_t>(run_cb));
  _testruns.push_back(std::make_shared<testrun_t>(run_ll));
  _testruns.push_back(std::make_shared<testrun_t>(run_kd));

  for (auto const& file : objects) {

    std::size_t nsurfaces = _objects.find(file)->second->size();

    for (unsigned id = 0; id != nsurfaces; ++id) {

      auto surface = _objects.find(file)->second->begin();
      std::advance(surface, id);

      auto domain = (**surface).domain();

      _testruns[0]->data.push_back({ file, id, domain, 0, 0.0 });
      _testruns[1]->data.push_back({ file, id, domain, 0, 0.0 });
      _testruns[2]->data.push_back({ file, id, domain, 0, 0.0 });
      _testruns[3]->data.push_back({ file, id, domain, 0, 0.0 });
    }
  } 
#else
  mainwindow* win = dynamic_cast<mainwindow*>(parent());
  if (!win) {
    throw std::runtime_error("No Main Window");
  }

  testrun_t run = { _view, _aamode, 0U, _current_texresolution, _tex_classification_enabled, {}, 0.0 };
  _testruns.push_back(std::make_shared<testrun_t>(run));

  for (auto const& file : objects) {

    std::size_t nsurfaces = _objects.find(file)->second->size();

    for (unsigned id = 0; id != nsurfaces; ++id) {
      auto surface = _objects.find(file)->second->begin();
      std::advance(surface, id);

      auto domain = (**surface).domain();
      _testruns[0]->data.push_back({ file, id, domain, 0, 0.0 });
    }
  }
#endif
}

///////////////////////////////////////////////////////////////////////
void                    
glwidget::texture_filtering(int enable)
{
  _linear_filter = enable != 0;
}

///////////////////////////////////////////////////////////////////////
void 
glwidget::resizeGL(int width, int height)
{
  _width  = width;
  _height = height;

  glViewport(0, 0, _width, _height);
}


///////////////////////////////////////////////////////////////////////
void 
glwidget::paintGL()
{
  /////////////////////////////////////////
  // initialize renderer
  /////////////////////////////////////////
  if ( !_initialized )
  {
    _init();
    resizeGL(_width, _height);
  }

  glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
  glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

  if (!_gputimer) {
    std::cout << "create timer query" << std::endl;
    _gputimer.reset(new gpucast::gl::timer_query);
  }

  /////////////////////////////////////////////////////////////////////////////
  // test setup
  /////////////////////////////////////////////////////////////////////////////
  if (!_testruns.empty()) {
    auto run = _testruns.back();
    if (run->current == run->data.size()) {
      run->save();
      _testruns.pop_back();
    }
    else{
      auto id = run->current;
      std::cout << "progress : " << float(100*id) / run->data.size() << "%             \r";
      update_view(run->data[id].filename, run->data[id].index, run->rendermode, run->antialiasing);
    }
  }
  else {
    mainwindow* win = dynamic_cast<mainwindow*>(parent());
    if (win) {
      win->enable();
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  // test setup
  /////////////////////////////////////////////////////////////////////////////

  _gputimer->begin();

  auto view_center = gpucast::math::vec2f(_domain_min[0] + _shift_x*_domain_size[0], _domain_min[1] + _shift_y*_domain_size[1]);
  auto view_size = gpucast::math::vec2f(_domain_size[0], _domain_size[1]);
  auto view_shift = gpucast::math::vec2f(_shift_x, _shift_y);

  float offset_x = _domain_min[0];
  float offset_y = _domain_min[1];

  switch ( _view )
  {
    // simply draw gl lines
    case original :
    case double_binary_partition :
    case minification:
    case contour_map_binary_partition :
    case contour_map_kd_partition:
    case contour_map_loop_list_partition :
      { 
        _partition_program->begin();
        _partition_program->set_uniform_matrix4fv("mvp", 1, false, &_projection[0]); 
        _partition_program->set_uniform2f("domain_min", view_center[0], view_center[1]);
        _partition_program->set_uniform2f("domain_base", _domain_min[0], _domain_min[1]);
        _partition_program->set_uniform1f("domain_zoom", _zoom);
        _partition_program->set_uniform1i("pixelsize", _pixel_size);
        std::for_each ( _curve_geometry.begin(), _curve_geometry.end(), std::bind(&gpucast::gl::line::draw, std::placeholders::_1, 1.0f));
        _partition_program->end();
        break;
      }
    // per-pixel classification
    case double_binary_classification : 
      {
        _db_program->begin();
        {
          _db_program->set_uniform1i ( "trimid", _trimid );
          _db_program->set_uniform1i ( "width",  _width );
          _db_program->set_uniform1i ( "height", _height );
          _db_program->set_uniform1i ( "show_costs", int(_show_texel_fetches) );
          _db_program->set_uniform1i ( "antialiasing", int(_aamode) );
                                                  
          _db_program->set_uniform2f("domain_size", view_size[0], view_size[1]);
          _db_program->set_uniform2f ("domain_min", view_center[0], view_center[1]);
          _db_program->set_uniform1f("domain_zoom", _zoom);

          unsigned texunit = 0;
          _db_program->set_texturebuffer ( "trimdata",         *_db_trimdata,        texunit++ );
          _db_program->set_texturebuffer ( "celldata",         *_db_celldata,        texunit++ );
          _db_program->set_texturebuffer ( "curvelist",        *_db_curvelist,       texunit++ );
          _db_program->set_texturebuffer ( "curvedata",        *_db_curvedata,       texunit++ );
          _db_program->set_texture1d     ( "transfertexture",  *_transfertexture,    texunit++ );
          _db_program->set_texturebuffer ( "sampler_preclass", *_db_preclassdata,    texunit++ );
          _db_program->set_uniform1i("pixelsize", _pixel_size);

          _bilinear_filter->bind(texunit);
          _db_program->set_texture2d("prefilter_texture", *_prefilter_texture, texunit++);

          _quad->draw();
        }
        _db_program->end();

        if (!_testruns.empty()) {
          _partition_program->begin();
          _partition_program->set_uniform_matrix4fv("mvp", 1, false, &_projection[0]);
          _partition_program->set_uniform2f("domain_base", _domain_min[0], _domain_min[1]);
          _partition_program->set_uniform2f("domain_min", _domain_size[0], _domain_size[1]);
          _partition_program->set_uniform2f("view_min", view_center[0], view_center[1]);
          _partition_program->set_uniform1f("domain_zoom", _zoom);
          _partition_program->set_uniform1i("pixelsize", _pixel_size);
          std::for_each(_curve_geometry.begin(), _curve_geometry.end(), std::bind(&gpucast::gl::line::draw, std::placeholders::_1, 1.0f));
          _partition_program->end();
        }
        break;
      }
      // per-pixel classification contour search
    case contour_map_kd_classification:
      _kd_program->begin();
      {
        _kd_program->set_uniform1i("trimid", _trimid);
        _kd_program->set_uniform1i("width", _width);
        _kd_program->set_uniform1i("height", _height);
        _kd_program->set_uniform1i("show_costs", int(_show_texel_fetches));
        _kd_program->set_uniform1i("antialiasing", int(_aamode));
        _kd_program->set_uniform2f("domain_size", view_size[0], view_size[1]);
        _kd_program->set_uniform2f("domain_min", view_center[0], view_center[1]);

        _kd_program->set_uniform1f("domain_zoom", _zoom);

        unsigned texunit = 0;
        _kd_program->set_texturebuffer("sampler_partition", *_kd_partition, texunit++);
        _kd_program->set_texturebuffer("sampler_contourlist", *_kd_contourlist, texunit++);
        _kd_program->set_texturebuffer("sampler_curvelist", *_kd_curvelist, texunit++);
        _kd_program->set_texturebuffer("sampler_curvedata", *_kd_curvedata, texunit++);
        _kd_program->set_texturebuffer("sampler_pointdata", *_kd_pointdata, texunit++);
        _kd_program->set_texturebuffer("sampler_preclass", *_kd_preclassdata, texunit++);
        _kd_program->set_texture1d("transfertexture", *_transfertexture, texunit++);
        _kd_program->set_uniform1i("pixelsize", _pixel_size);

        _bilinear_filter->bind(texunit);
        _kd_program->set_texture2d("prefilter_texture", *_prefilter_texture, texunit++);

        _quad->draw();

      }
      _kd_program->end();

      if (!_testruns.empty()) {
        _partition_program->begin();
        _partition_program->set_uniform_matrix4fv("mvp", 1, false, &_projection[0]);
        _partition_program->set_uniform2f("domain_base", _domain_min[0], _domain_min[1]);
        _partition_program->set_uniform2f("domain_min", view_center[0], view_center[1]);
        _partition_program->set_uniform1f("domain_zoom", _zoom);
        _partition_program->set_uniform1i("pixelsize", _pixel_size);
        std::for_each(_curve_geometry.begin(), _curve_geometry.end(), std::bind(&gpucast::gl::line::draw, std::placeholders::_1, 1.0f));
        _partition_program->end();
      }
      break;

    case contour_map_binary_classification :
            
        _cmb_program->begin();
        {
          _cmb_program->set_uniform1i ( "trimid", _trimid );
          _cmb_program->set_uniform1i ( "width",  _width );
          _cmb_program->set_uniform1i ( "height", _height );
          _cmb_program->set_uniform1i ( "show_costs", int(_show_texel_fetches) );
          _cmb_program->set_uniform1i("antialiasing", int(_aamode));
                                                  
          _cmb_program->set_uniform2f("domain_size", view_size[0], view_size[1]);
          _cmb_program->set_uniform2f("domain_min", view_center[0], view_center[1]);
          _cmb_program->set_uniform1f("domain_zoom", _zoom);

          unsigned texunit = 0;
          _cmb_program->set_texturebuffer ( "sampler_partition",   *_cmb_partition,  texunit++ );
          _cmb_program->set_texturebuffer ( "sampler_contourlist", *_cmb_contourlist, texunit++ );
          _cmb_program->set_texturebuffer ( "sampler_curvelist",   *_cmb_curvelist,   texunit++ );
          _cmb_program->set_texturebuffer ( "sampler_curvedata",   *_cmb_curvedata,   texunit++ );
          _cmb_program->set_texturebuffer ( "sampler_pointdata",   *_cmb_pointdata,   texunit++ );
          _cmb_program->set_texture1d     ( "transfertexture",     *_transfertexture, texunit++ );
          _cmb_program->set_texturebuffer ("sampler_preclass",     *_cmb_preclassdata, texunit++);
          _cmb_program->set_uniform1i("pixelsize", _pixel_size);

          _bilinear_filter->bind(texunit);
          _cmb_program->set_texture2d("prefilter_texture", *_prefilter_texture, texunit++);

          _quad->draw();
        }
        _cmb_program->end();

        if (!_testruns.empty()) {
          _partition_program->begin();
          _partition_program->set_uniform_matrix4fv("mvp", 1, false, &_projection[0]);
          _partition_program->set_uniform2f("domain_base", _domain_min[0], _domain_min[1]);
          _partition_program->set_uniform2f("domain_min", view_center[0], view_center[1]);
          _partition_program->set_uniform1f("domain_zoom", _zoom);
          _partition_program->set_uniform1i("pixelsize", _pixel_size);
          std::for_each(_curve_geometry.begin(), _curve_geometry.end(), std::bind(&gpucast::gl::line::draw, std::placeholders::_1, 1.0f));
          _partition_program->end();
        }
        break;
    case contour_map_loop_list_classification:
      _loop_list_program->begin();
      {
        _loop_list_program->set_uniform1i("trim_index", int(_trimid));
        _loop_list_program->set_uniform1i("width", _width);
        _loop_list_program->set_uniform1i("height", _height);
        _loop_list_program->set_uniform1i("show_costs", int(_show_texel_fetches));
        _loop_list_program->set_uniform1i("antialiasing", int(_aamode));
        
        _loop_list_program->set_uniform2f("domain_size", view_size[0], view_size[1]);
        _loop_list_program->set_uniform2f("domain_min", view_center[0], view_center[1]);
        _loop_list_program->set_uniform1f("domain_zoom", _zoom);
        _loop_list_program->set_uniform1i("pixelsize", _pixel_size);

        _loop_list_loops->bind_buffer_base(0);
        _loop_list_program->set_shaderstoragebuffer("loop_buffer", *_loop_list_loops, 0);

        _loop_list_contours->bind_buffer_base(1);
        _loop_list_program->set_shaderstoragebuffer("contour_buffer", *_loop_list_contours, 1);

        _loop_list_curves->bind_buffer_base(2);
        _loop_list_program->set_shaderstoragebuffer("curve_buffer", *_loop_list_curves, 2);

        _loop_list_points->bind_buffer_base(3);
        _loop_list_program->set_shaderstoragebuffer("point_buffer", *_loop_list_points, 3);

        _bilinear_filter->bind(4);
        _loop_list_program->set_texture2d("prefilter_texture", *_prefilter_texture, 4);

        _loop_list_program->set_texturebuffer("sampler_preclass", *_loop_list_preclassdata, 5);

        _quad->draw();
      }
      _loop_list_program->end();

      if (!_testruns.empty()) {
        _partition_program->begin();
        _partition_program->set_uniform_matrix4fv("mvp", 1, false, &_projection[0]);
        _partition_program->set_uniform2f("domain_base", _domain_min[0], _domain_min[1]);
        _partition_program->set_uniform2f("domain_min", view_center[0], view_center[1]);
        _partition_program->set_uniform1f("domain_zoom", _zoom);
        _partition_program->set_uniform1i("pixelsize", _pixel_size);
        std::for_each(_curve_geometry.begin(), _curve_geometry.end(), std::bind(&gpucast::gl::line::draw, std::placeholders::_1, 1.0f));
        _partition_program->end();
      }
      break;

    case distance_field:
      _tex_program->begin();
      {
        if (_linear_filter) {
          _bilinear_filter->bind(0);
        }
        else {
          _nearest_filter->bind(0);
        }
        _tex_program->set_texture2d("classification_texture", *_distance_field_texture, 0);
        _tex_program->set_uniform1i("show_costs", int(_show_texel_fetches));
        _tex_program->set_uniform2f("domain_shift", view_shift[0], view_shift[1]);
        _tex_program->set_uniform1f("domain_zoom", _zoom);
        _tex_program->set_uniform1i("pixelsize", _pixel_size);
        _quad->draw();
      }
      _tex_program->end();

      _partition_program->begin();
      _partition_program->set_uniform_matrix4fv("mvp", 1, false, &_projection[0]);
      _partition_program->set_uniform2f("domain_base", _domain_min[0], _domain_min[1]);
      _partition_program->set_uniform2f("domain_min", view_center[0], view_center[1]);
      _partition_program->set_uniform1f("domain_zoom", _zoom);
      _partition_program->set_uniform1i("pixelsize", _pixel_size);
      std::for_each(_curve_geometry.begin(), _curve_geometry.end(), std::bind(&gpucast::gl::line::draw, std::placeholders::_1, 1.0f));
      _partition_program->end();

      break;

    case binary_field:
      _tex_program->begin(); 
      {
        if (_linear_filter) {
          _bilinear_filter->bind(0);
        }
        else {
          _nearest_filter->bind(0);
        }
        _tex_program->set_texture2d("classification_texture", *_binary_texture, 0);
        _tex_program->set_uniform2f("domain_shift", view_shift[0], view_shift[1]);
        _tex_program->set_uniform1f("domain_zoom", _zoom);
        _tex_program->set_uniform1i("pixelsize", _pixel_size);
        _tex_program->set_uniform1i("show_costs", int(_show_texel_fetches));

        _quad->draw();
      }
      _tex_program->end();

      _partition_program->begin();
      _partition_program->set_uniform_matrix4fv("mvp", 1, false, &_projection[0]);
      _partition_program->set_uniform2f("domain_base", _domain_min[0], _domain_min[1]);
      _partition_program->set_uniform2f("domain_min", view_center[0], view_center[1]);
      _partition_program->set_uniform1f("domain_zoom", _zoom);
      _partition_program->set_uniform1i("pixelsize", _pixel_size);
      std::for_each(_curve_geometry.begin(), _curve_geometry.end(), std::bind(&gpucast::gl::line::draw, std::placeholders::_1, 1.0f));
      _partition_program->end();
      break;
    case prefilter:
      _prefilter_program->begin();
      {
        if (_linear_filter) {
          _bilinear_filter->bind(0);
        }
        else {
          _nearest_filter->bind(0);
        }
        _prefilter_program->set_texture2d("prefilter_texture", *_prefilter_texture, 0);
        _quad->draw();
      }
      _prefilter_program->end();
    default : 
      break;
  };

  // end query
  _gputimer->end();

  // try to get query result
  double time_ms = _gputimer->time_in_ms();

  if (!_testruns.empty()) {
    _testruns.back()->data[_testruns.back()->current].time_ms = time_ms;
  }

  mainwindow* win = dynamic_cast<mainwindow*>(parent());
  if (win) {
    win->show_drawtime(time_ms);
  }

  if (!_testruns.empty()) {
    ++_testruns.back()->current;
  }

  // redraw
  this->update();
}


///////////////////////////////////////////////////////////////////////
/* virtual */ void  
glwidget::keyPressEvent ( QKeyEvent* /*event*/)
{}


///////////////////////////////////////////////////////////////////////
/* virtual */ void  
glwidget::keyReleaseEvent ( QKeyEvent* event )
{
  char key = event->key();

  if (event->modifiers() != Qt::ShiftModifier) {
    key = std::tolower(key);
  }

  switch ( key )
  {
    case 'r' : 
      {
        _initialize_shader();
        break;
      }
    default : break;
  };
}

///////////////////////////////////////////////////////////////////////
/* virtual */ void glwidget::mouseDoubleClickEvent(QMouseEvent * event)
{}

///////////////////////////////////////////////////////////////////////
/* virtual */ void glwidget::mouseMoveEvent(QMouseEvent * event)
{
  if ( _shift_mode ) {
    _shift_x += float(_last_x - event->x()) / _width;
    _shift_y -= float(_last_y - event->y()) / _height;
    _last_x = event->x();
    _last_y = event->y();
  }

  if (_zoom_mode) {
    _zoom *= float(_height - (_last_y - event->y())) / _height;
    _last_y = event->y();
  }
}

///////////////////////////////////////////////////////////////////////
/* virtual */ void glwidget::mousePressEvent(QMouseEvent * event)
{
  switch (event->button()) {
    case Qt::LeftButton : 
    case Qt::MidButton:
      _shift_mode = true;
      _zoom_mode = false;
      _last_x = event->x();
      _last_y = event->y();
      break;
    case Qt::RightButton : 
      _zoom_mode = true;
      _shift_mode = false;
      _last_x = event->x();
      _last_y = event->y();
      break;
  }
}

///////////////////////////////////////////////////////////////////////
/* virtual */ void glwidget::mouseReleaseEvent(QMouseEvent * event)
{
  _zoom_mode = false;
  _shift_mode = false;
}

///////////////////////////////////////////////////////////////////////
void
glwidget::_init()
{
  if ( _initialized ) 
  {
    return;
  } else {
    _initialized = true;
  }

  mainwindow* mainwin = dynamic_cast<mainwindow*>(parent());
  if ( mainwin )
  {
    mainwin->show();
  } else {
    _initialized = false;
    return;
  }

  gpucast::gl::init_glew ( std::cout );

  gpucast::gl::print_contextinfo     ( std::cout );

  gpucast::gl::print_extensions      ( std::cout );

  _initialize_shader();

  glEnable ( GL_DEPTH_TEST );

  if ( !_quad ) {
    _quad = make_unique<gpucast::gl::plane>(0, -1, -1);
  }

  if ( !_transfertexture )
  {
    _transfertexture = make_unique<gpucast::gl::texture1d>();
    gpucast::gl::transferfunction<gpucast::math::vec4f> tf;
    tf.set ( 0, gpucast::math::vec4f(0.0f, 0.0f, 0.5f, 1.0f) );
    tf.set ( 60, gpucast::math::vec4f(0.0f, 0.7f, 0.5f, 1.0f) );
    tf.set ( 120, gpucast::math::vec4f(0.0f, 0.9f, 0.0f, 1.0f) );
    tf.set ( 180, gpucast::math::vec4f(0.9f, 0.9f, 0.0f, 1.0f) );
    tf.set ( 255, gpucast::math::vec4f(0.9f, 0.0f, 0.0f, 1.0f) );

    std::vector<gpucast::math::vec4f> samples;
    tf.evaluate(256, samples, gpucast::gl::piecewise_linear());
    _transfertexture->teximage(0, GL_RGBA, 256, 0, GL_RGBA, GL_FLOAT, &samples.front()[0] );
    _transfertexture->set_parameteri ( GL_TEXTURE_MAG_FILTER, GL_LINEAR );
    _transfertexture->set_parameteri ( GL_TEXTURE_MIN_FILTER, GL_LINEAR );
  }
}


///////////////////////////////////////////////////////////////////////
void 
glwidget::_initialize_shader()  
{
  gpucast::gl::resource_factory program_factory;

  _partition_program = program_factory.create_program({ 
    { gpucast::gl::vertex_stage, "./shader/partition.vert.glsl"}, 
    { gpucast::gl::fragment_stage, "./shader/partition.frag.glsl"}
  });

  _db_program = program_factory.create_program({ 
    { gpucast::gl::vertex_stage, "./shader/double_binary.vert" },
    { gpucast::gl::fragment_stage, "./shader/double_binary.frag"}
  });

  _tex_program = program_factory.create_program({ 
    { gpucast::gl::vertex_stage, "./shader/texture_view.vert.glsl" },
    { gpucast::gl::fragment_stage, "./shader/texture_view.frag.glsl"}
  });

  _cmb_program = program_factory.create_program({
    { gpucast::gl::vertex_stage, "./shader/contour_binary.vert" },
    { gpucast::gl::fragment_stage, "./shader/contour_binary.frag"}
  });

  _kd_program = program_factory.create_program({
    { gpucast::gl::vertex_stage, "./shader/contour_kd.vert" },
    { gpucast::gl::fragment_stage, "./shader/contour_kd.frag"}
  });

  _loop_list_program = program_factory.create_program({
    { gpucast::gl::vertex_stage, "./shader/contour_loop_list.vert"},
    { gpucast::gl::fragment_stage, "./shader/contour_loop_list.frag"}
  });

  _prefilter_program = program_factory.create_program({
    { gpucast::gl::vertex_stage, "./shader/prefilter_view.vert.glsl"},
    { gpucast::gl::fragment_stage, "./shader/prefilter_view.frag.glsl"}
  });
}

///////////////////////////////////////////////////////////////////////
void glwidget::_initialize_prefilter()
{
  if (!_prefilter_texture) {

    _prefilter_texture = make_unique<gpucast::gl::texture2d>();

    gpucast::math::util::prefilter2d<gpucast::math::vec2d> pre_integrator(32, 0.5);

    unsigned prefilter_resolution = 64;
    std::vector<float> texture_data;

    auto distance_offset = std::sqrt(2) / prefilter_resolution;
    auto angle_offset = (2.0 * M_PI) / prefilter_resolution;

    for (unsigned d = 0; d != prefilter_resolution; ++d) {
      for (unsigned a = 0; a != prefilter_resolution; ++a) {

        auto angle = a * angle_offset;
        auto distance = -1.0 / std::sqrt(2) + distance_offset * d;
        auto alpha = pre_integrator(gpucast::math::vec2d(angle, distance));

        texture_data.push_back(alpha);
      }
    }

    _prefilter_texture->teximage(0, GL_R32F, prefilter_resolution, prefilter_resolution, 0, GL_RED, GL_FLOAT, (void*)(&texture_data[0]));
  }
}