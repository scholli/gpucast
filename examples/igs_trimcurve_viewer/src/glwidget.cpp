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

#include <gpucast/core/config.hpp>
#include <gpucast/gl/vertexshader.hpp>
#include <gpucast/gl/elementarraybuffer.hpp>
#include <gpucast/gl/arraybuffer.hpp>
#include <gpucast/math/vec4.hpp>
#include <gpucast/gl/fragmentshader.hpp>
#include <gpucast/gl/error.hpp>
#include <gpucast/gl/util/transferfunction.hpp>
#include <gpucast/gl/util/resource_factory.hpp>

#include <boost/filesystem.hpp>
#include <boost/unordered_map.hpp>

#include <gpucast/core/surface_converter.hpp>
#include <gpucast/core/import/igs.hpp>
#include <gpucast/core/trimdomain.hpp>
#include <gpucast/core/trimdomain_serializer_double_binary.hpp>
#include <gpucast/core/trimdomain_serializer_contour_map_binary.hpp>
#include <gpucast/core/trimdomain_serializer_contour_map_kd.hpp>
#include <gpucast/core/trimdomain_serializer_loop_contour_list.hpp>


#include <gpucast/math/parametric/domain/partition/monotonic_contour/contour_map_binary.hpp>
#include <gpucast/math/parametric/domain/partition/monotonic_contour/contour_map_loop_list.hpp>
#include <gpucast/math/parametric/domain/partition/double_binary/partition.hpp>


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
    std::shared_ptr<gpucast::nurbssurfaceobject>  nobj = igs_loader.load(filename);
    std::shared_ptr<gpucast::beziersurfaceobject> bobj = std::make_shared<gpucast::beziersurfaceobject>();

    gpucast::surface_converter converter;
    converter.convert(nobj, bobj);
    _objects[filename] = bobj;
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
glwidget::update_view(std::string const& name, std::size_t const index, view current_view, unsigned resolution)
{
  _current_object  = name;
  _current_surface = index;
  _view            = current_view;

  clear();

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

      switch ( _view )
      {
        case original :
          generate_original_view ( domain );
          break;
        case double_binary_partition :
          generate_double_binary_view ( domain );
          break;
        case double_binary_classification :
          serialize_double_binary ( domain );
          break;
        case contour_map_binary_partition :
          generate_bboxmap_view ( domain );
          break;
        case contour_map_binary_classification :
          serialize_contour_binary ( domain );
          break;
        case minification:
          generate_minification_view(domain, resolution);
          break;
        case contour_map_loop_list_partition:
          generate_loop_list_view(domain);
          break;
        case contour_map_loop_list_classification:
          serialize_contour_loop_list(domain);
          break;
        case distance_field:
          generate_distance_field(domain, resolution);
          break;
        case binary_field:
          generate_binary_field(domain, resolution);
          break;
        default : 
          break;
      };

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

  gpucast::math::domain::partition<gpucast::beziersurface::curve_point_type>  partition(domain->curves().begin(), domain->curves().end());
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

  for (auto const& loop : domain->loops())
  {
    looplist.add(gpucast::math::domain::contour<double>(loop.begin(), loop.end()));
  }

  looplist.initialize();

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
glwidget::generate_minification_view(gpucast::beziersurface::trimdomain_ptr const& domain, unsigned resolution)
{
  generate_trim_region_vbo(domain);

  unsigned res_u = resolution;
  unsigned res_v = resolution;

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

      unsigned intersections = 0;
      float minimal_distance = std::numeric_limits<float>::max();

      for (auto const& loop : domain->loops()) {
        intersections += loop.is_inside(ubox.center());

        for (auto const& curve : loop.curves()) {
          if (!_optimal_distance) {
            minimal_distance = std::min(float(curve->estimate_closest_distance(ubox.center())), minimal_distance);
          }
          else {
            minimal_distance = std::min(float(curve->closest_distance(ubox.center())), minimal_distance);
          }
        }
      }

      float intensity = 0.1f + minimal_distance / std::max(offset_u, offset_v);
      gpucast::math::vec4f color_inside(0.0f, intensity, 0.0f, 1.0f);
      gpucast::math::vec4f color_outside(intensity, 0.0f, 0.0f, 1.0f);

      if (intersections % 2) { 
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

  initialize_filter();

  _binary_texture = std::make_unique<gpucast::gl::texture2d>();
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

  initialize_filter();

  _distance_field_texture = std::make_unique<gpucast::gl::texture2d>();
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
      float minimal_distance = std::numeric_limits<float>::max();

      for (auto const& loop : domain->loops()) {
        // find out interections ->
        intersections += loop.is_inside(ubox.center());
        for (auto const& curve : loop.curves()) 
        {
          if (curve->bbox_simple().distance(ubox.center()) < minimal_distance) 
          {
            if (!_optimal_distance) {
              minimal_distance = std::min(float(curve->estimate_closest_distance(ubox.center())), minimal_distance);
            }
            else {
              minimal_distance = std::min(float(curve->closest_distance(ubox.center())), minimal_distance);
            }
          }
        }
      }

      if (intersections % 2) {
        texture_data.push_back(1.0f * minimal_distance );
      }
      else {
        texture_data.push_back(-1.0f * minimal_distance );
      }
    }
  }

  _distance_field_texture->teximage(0, GL_R32F, res_u, res_v, 0, GL_RED, GL_FLOAT, (void*)(&texture_data[0]));
}


///////////////////////////////////////////////////////////////////////
void                    
glwidget::serialize_double_binary ( gpucast::beziersurface::trimdomain_ptr const& domain )
{
  if ( !_db_trimdata )   _db_trimdata  = std::make_unique<gpucast::gl::texturebuffer>();
  if ( !_db_celldata )   _db_celldata  = std::make_unique<gpucast::gl::texturebuffer>();
  if ( !_db_curvelist )  _db_curvelist = std::make_unique<gpucast::gl::texturebuffer>();
  if ( !_db_curvedata )  _db_curvedata = std::make_unique<gpucast::gl::texturebuffer>();

  std::unordered_map<gpucast::trimdomain::curve_ptr, gpucast::trimdomain_serializer::address_type>          referenced_curves;
  std::unordered_map<gpucast::beziersurface::trimdomain_ptr, gpucast::trimdomain_serializer::address_type>  referenced_domains;

  std::vector<gpucast::math::vec4f> trimdata(1);
  std::vector<gpucast::math::vec4f> celldata(1);
  std::vector<gpucast::math::vec4f> curvelists(1);
  std::vector<gpucast::math::vec3f> curvedata(1);

  gpucast::trimdomain_serializer_double_binary serializer;
  _trimid = serializer.serialize ( domain, referenced_domains, referenced_curves, trimdata, celldata, curvelists, curvedata ) ;

  _db_trimdata->update  ( trimdata.begin(), trimdata.end() );
  _db_celldata->update  ( celldata.begin(), celldata.end() );
  _db_curvelist->update ( curvelists.begin(), curvelists.end() );
  _db_curvedata->update ( curvedata.begin(), curvedata.end() );
                       
  _db_trimdata->format  ( GL_RGBA32F );
  _db_celldata->format  ( GL_RGBA32F );
  _db_curvelist->format ( GL_RGBA32F );
  _db_curvedata->format ( GL_RGB32F );

  _domain_size = gpucast::math::vec2f ( domain->nurbsdomain().size() );
  _domain_min  = gpucast::math::vec2f ( domain->nurbsdomain().min );

  // show memory usage
  std::size_t size_bytes = ((trimdata.size() - 1)   * sizeof(gpucast::math::vec4f) +
    (celldata.size() - 1)   * sizeof(gpucast::math::vec4f) +
    (curvelists.size() - 1) * sizeof(gpucast::math::vec4f) +
    (curvedata.size() - 1)  * sizeof(gpucast::math::vec3f));

  mainwindow* win = dynamic_cast<mainwindow*>(parent());
  if (win) {
    win->show_memusage(size_bytes);
  }
}

///////////////////////////////////////////////////////////////////////
void                    
glwidget::serialize_contour_binary ( gpucast::beziersurface::trimdomain_ptr const& domain )
{
  if (!_cmb_partition)   _cmb_partition = std::make_unique<gpucast::gl::texturebuffer>();
  if (!_cmb_contourlist) _cmb_contourlist = std::make_unique<gpucast::gl::texturebuffer>();
  if (!_cmb_curvelist) _cmb_curvelist = std::make_unique<gpucast::gl::texturebuffer>();
  if (!_cmb_curvedata) _cmb_curvedata = std::make_unique<gpucast::gl::texturebuffer>();
  if (!_cmb_pointdata) _cmb_pointdata = std::make_unique<gpucast::gl::texturebuffer>();

  std::unordered_map<gpucast::trimdomain::curve_ptr,         gpucast::trimdomain_serializer::address_type>                                  referenced_curves;
  std::unordered_map<gpucast::beziersurface::trimdomain_ptr, gpucast::trimdomain_serializer::address_type>                                  referenced_domains;
  std::unordered_map<gpucast::trimdomain_serializer_contour_map_binary::contour_segment_ptr, gpucast::trimdomain_serializer::address_type>  referenced_contours;

  std::vector<gpucast::math::vec4f> partition(1);
  std::vector<gpucast::math::vec2f> contourlist(1);
  std::vector<gpucast::math::vec4f> curvelist(1);
  std::vector<float> curvedata(1);
  std::vector<gpucast::math::vec3f> pointdata(1);

  gpucast::trimdomain_serializer_contour_map_binary serializer;
  _trimid = serializer.serialize ( domain, 
                                   referenced_domains, 
                                   referenced_curves, 
                                   referenced_contours, 
                                   partition, 
                                   contourlist,
                                   curvelist,
                                   curvedata,
                                   pointdata );

  _cmb_partition->update   ( partition.begin(), partition.end());
  _cmb_contourlist->update ( contourlist.begin(), contourlist.end());
  _cmb_curvelist->update   ( curvelist.begin(), curvelist.end());
  _cmb_curvedata->update   ( curvedata.begin(), curvedata.end());
  _cmb_pointdata->update   ( pointdata.begin(), pointdata.end());

  _cmb_partition->format   ( GL_RGBA32F );
  _cmb_contourlist->format ( GL_RG32F );
  _cmb_curvelist->format   ( GL_RGBA32F );
  _cmb_curvedata->format   ( GL_R32F );
  _cmb_pointdata->format   ( GL_RGB32F );

  _domain_size = gpucast::math::vec2f ( domain->nurbsdomain().size() );
  _domain_min  = gpucast::math::vec2f ( domain->nurbsdomain().min );

  // show memory usage
  std::size_t size_bytes = ((partition.size() - 1) * sizeof(gpucast::math::vec4f) +
    (contourlist.size() - 1) * sizeof(gpucast::math::vec2f) +
    (curvelist.size() - 1) * sizeof(gpucast::math::vec4f) +
    (curvedata.size() - 1) * sizeof(float)+
    (pointdata.size() - 1) * sizeof(gpucast::math::vec3f));

  mainwindow* win = dynamic_cast<mainwindow*>(parent());
  if (win) {
    win->show_memusage(size_bytes);
  }
}


///////////////////////////////////////////////////////////////////////
void
glwidget::serialize_contour_loop_list(gpucast::beziersurface::trimdomain_ptr const& domain)
{
  if (!_loop_list_loops)    _loop_list_loops = std::make_unique<gpucast::gl::shaderstoragebuffer>();
  if (!_loop_list_contours) _loop_list_contours = std::make_unique<gpucast::gl::shaderstoragebuffer>();
  if (!_loop_list_curves)   _loop_list_curves = std::make_unique<gpucast::gl::shaderstoragebuffer>();
  if (!_loop_list_points)   _loop_list_points = std::make_unique<gpucast::gl::shaderstoragebuffer>();

  gpucast::trimdomain_serializer_loop_contour_list::serialization serialization;
  gpucast::trimdomain_serializer_loop_contour_list serializer;

  std::unordered_map<gpucast::beziersurface::trimdomain_ptr, gpucast::trimdomain_serializer::address_type> referenced_domains;

  _trimid = serializer.serialize(domain, referenced_domains, serialization);

  // write data to shader storage
  _loop_list_loops->update(serialization.loops.begin(), serialization.loops.end());
  _loop_list_contours->update(serialization.contours.begin(), serialization.contours.end());
  _loop_list_curves->update(serialization.curves.begin(), serialization.curves.end());
  _loop_list_points->update(serialization.points.begin(), serialization.points.end());

  _domain_size = gpucast::math::vec2f(domain->nurbsdomain().size());
  _domain_min = gpucast::math::vec2f(domain->nurbsdomain().min);

  // show memory usage
  std::size_t size_bytes = ((serialization.loops.size() - 1) * sizeof(gpucast::trimdomain_serializer_loop_contour_list::serialization::loop_t) +
    (serialization.contours.size() - 1) * sizeof(gpucast::trimdomain_serializer_loop_contour_list::serialization::contour_t) +
    (serialization.curves.size() - 1) * sizeof(gpucast::trimdomain_serializer_loop_contour_list::serialization::curve_t) +
    (serialization.points.size() - 1) * sizeof(gpucast::trimdomain_serializer_loop_contour_list::serialization::point_t));

  mainwindow* win = dynamic_cast<mainwindow*>(parent());
  if (win) {
    win->show_memusage(size_bytes);
  }
}

///////////////////////////////////////////////////////////////////////
void glwidget::initialize_filter()
{
  _bilinear_filter = std::make_unique<gpucast::gl::sampler>();
  _bilinear_filter->parameter(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  _bilinear_filter->parameter(GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  _bilinear_filter->parameter(GL_TEXTURE_WRAP_R, GL_CLAMP);
  _bilinear_filter->parameter(GL_TEXTURE_WRAP_S, GL_CLAMP);
  _bilinear_filter->parameter(GL_TEXTURE_WRAP_T, GL_CLAMP);
                              
  _nearest_filter = std::make_unique<gpucast::gl::sampler>();
  _nearest_filter->parameter(GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  _nearest_filter->parameter(GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  _nearest_filter->parameter(GL_TEXTURE_WRAP_R, GL_CLAMP);
  _nearest_filter->parameter(GL_TEXTURE_WRAP_S, GL_CLAMP);
  _nearest_filter->parameter(GL_TEXTURE_WRAP_T, GL_CLAMP);
}

///////////////////////////////////////////////////////////////////////
void glwidget::generate_trim_region_vbo(gpucast::beziersurface::trimdomain_ptr const& domain)
{
  gpucast::trimdomain::curve_container curves = domain->curves();
  for (auto curve = curves.begin(); curve != curves.end(); ++curve)
  {
    gpucast::math::vec4f cpolygon_color(1.0f, 1.0f, 1.0f, 1.0f);
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
glwidget::add_gl_bbox (  gpucast::math::bbox2d const& bbox, gpucast::math::vec4f const& color )
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
  bbox_points.push_back(gpucast::math::vec4f(float(bbox.min[0]), float(bbox.min[1]), -2.0f, 1.0f));
  bbox_points.push_back(gpucast::math::vec4f(float(bbox.max[0]), float(bbox.max[1]), -2.0f, 1.0f));
  bbox_points.push_back(gpucast::math::vec4f(float(bbox.min[0]), float(bbox.max[1]), -2.0f, 1.0f));
  bbox_points.push_back(gpucast::math::vec4f(float(bbox.max[0]), float(bbox.min[1]), -2.0f, 1.0f));

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
glwidget::show_texel_fetches(int enable)
{
  _show_texel_fetches = enable != 0;
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
glwidget::recompile()
{
  std::cout << "Recompiling shader code." << std::endl;
  _initialize_shader();
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

  gpucast::gl::timer gtimer;
  gtimer.start();

  switch ( _view )
  {
    // simply draw gl lines
    case original :
    case double_binary_partition :
    case minification:
    case contour_map_binary_partition :
    case contour_map_loop_list_partition :
      {
        _partition_program->begin();
        _partition_program->set_uniform_matrix4fv("mvp", 1, false, &_projection[0]); 
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
                                                  
          _db_program->set_uniform2f ( "domain_size", _domain_size[0], _domain_size[1] );
          _db_program->set_uniform2f ( "domain_min",  _domain_min[0], _domain_min[1] );

          unsigned texunit = 0;
          _db_program->set_texturebuffer ( "trimdata",         *_db_trimdata,        texunit++ );
          _db_program->set_texturebuffer ( "celldata",         *_db_celldata,        texunit++ );
          _db_program->set_texturebuffer ( "curvelist",        *_db_curvelist,       texunit++ );
          _db_program->set_texturebuffer ( "curvedata",        *_db_curvedata,       texunit++ );
          _db_program->set_texture1d     ( "transfertexture",  *_transfertexture, texunit++ );

          _quad->draw();
        }
        _db_program->end();
        break;
      }
      // per-pixel classification contour search
    case contour_map_binary_classification :
            
        _cmb_program->begin();
        {
          _cmb_program->set_uniform1i ( "trimid", _trimid );
          _cmb_program->set_uniform1i ( "width",  _width );
          _cmb_program->set_uniform1i ( "height", _height );
          _cmb_program->set_uniform1i ( "show_costs", int(_show_texel_fetches) );
                                                  
          _cmb_program->set_uniform2f ( "domain_size", _domain_size[0], _domain_size[1] );
          _cmb_program->set_uniform2f ( "domain_min",  _domain_min[0], _domain_min[1] );

          unsigned texunit = 0;
          _cmb_program->set_texturebuffer ( "sampler_partition",   *_cmb_partition,  texunit++ );
          _cmb_program->set_texturebuffer ( "sampler_contourlist", *_cmb_contourlist, texunit++ );
          _cmb_program->set_texturebuffer ( "sampler_curvelist",   *_cmb_curvelist,   texunit++ );
          _cmb_program->set_texturebuffer ( "sampler_curvedata",   *_cmb_curvedata,   texunit++ );
          _cmb_program->set_texturebuffer ( "sampler_pointdata",   *_cmb_pointdata,   texunit++ );
          _cmb_program->set_texture1d     ( "transfertexture",     *_transfertexture, texunit++ );

          _quad->draw();
        }
        _cmb_program->end();
        break;
    case contour_map_loop_list_classification:
      _loop_list_program->begin();
      {
        _loop_list_program->set_uniform1i("trim_index", int(_trimid));
        _loop_list_program->set_uniform1i("width", _width);
        _loop_list_program->set_uniform1i("height", _height);
        _loop_list_program->set_uniform1i("show_costs", int(_show_texel_fetches));

        _loop_list_program->set_uniform2f("domain_size", _domain_size[0], _domain_size[1]);
        _loop_list_program->set_uniform2f("domain_min", _domain_min[0], _domain_min[1]);

        _loop_list_loops->bind_buffer_base(0);
        _loop_list_program->set_shaderstoragebuffer("loop_buffer", *_loop_list_loops, 0);

        _loop_list_contours->bind_buffer_base(1);
        _loop_list_program->set_shaderstoragebuffer("contour_buffer", *_loop_list_contours, 1);

        _loop_list_curves->bind_buffer_base(2);
        _loop_list_program->set_shaderstoragebuffer("curve_buffer", *_loop_list_curves, 2);

        _loop_list_points->bind_buffer_base(3);
        _loop_list_program->set_shaderstoragebuffer("point_buffer", *_loop_list_points, 3);

        _quad->draw();
      }
      _loop_list_program->end();
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
        _tex_program->set_uniform2f("domain_size", _domain_size[0], _domain_size[1]);
        _tex_program->set_uniform2f("domain_min", _domain_min[0], _domain_min[1]);

        _quad->draw();
      }
      _tex_program->end();

      _partition_program->begin();
      _partition_program->set_uniform_matrix4fv("mvp", 1, false, &_projection[0]);
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
        _tex_program->set_uniform1i("show_costs", int(_show_texel_fetches));

        _quad->draw();
      }
      _tex_program->end();

      _partition_program->begin();
      _partition_program->set_uniform_matrix4fv("mvp", 1, false, &_projection[0]);
      std::for_each(_curve_geometry.begin(), _curve_geometry.end(), std::bind(&gpucast::gl::line::draw, std::placeholders::_1, 1.0f));
      _partition_program->end();
      break;
    default : 
      break;
  };

  glFinish();
  gtimer.stop();
  double drawtime_ms = gtimer.result().as_seconds() * 1000.0;

  mainwindow* win = dynamic_cast<mainwindow*>(parent());
  if ( win )
  {
    win->show_drawtime(drawtime_ms);
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
    _quad = std::make_unique<gpucast::gl::plane>(0, -1, -1);
  }

  if ( !_transfertexture )
  {
    _transfertexture = std::make_unique<gpucast::gl::texture1d>();
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

  _partition_program = program_factory.create_program("./shader/partition.vert.glsl", "./shader/partition.frag.glsl");
  _db_program = program_factory.create_program("./shader/double_binary.vert", "./shader/double_binary.frag");
  _tex_program = program_factory.create_program("./shader/texture_view.vert.glsl", "./shader/texture_view.frag.glsl");
  _cmb_program = program_factory.create_program("./shader/contour_binary.vert", "./shader/contour_binary.frag");
  _kd_program = program_factory.create_program("./shader/domain.vert.glsl", "./shader/contour.frag.glsl");
  _loop_list_program = program_factory.create_program("./shader/contour_loop_list.vert", "./shader/contour_loop_list.frag");
}


