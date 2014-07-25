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

#include <gpucast/gl/vertexshader.hpp>
#include <gpucast/gl/elementarraybuffer.hpp>
#include <gpucast/gl/arraybuffer.hpp>
#include <gpucast/gl/math/vec4.hpp>
#include <gpucast/gl/fragmentshader.hpp>
#include <gpucast/gl/error.hpp>
#include <gpucast/gl/util/transferfunction.hpp>

#include <boost/filesystem.hpp>
#include <boost/unordered_map.hpp>

#include <gpucast/core/surface_converter.hpp>
#include <gpucast/core/import/igs.hpp>
#include <gpucast/core/trimdomain.hpp>
#include <gpucast/core/trimdomain_serializer_double_binary.hpp>
#include <gpucast/core/trimdomain_serializer_contour_map_binary.hpp>
#include <gpucast/core/trimdomain_serializer_contour_map_kd.hpp>

#include <gpucast/math/parametric/domain/contour_map_binary.hpp>
#include <gpucast/math/parametric/domain/partition.hpp>


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
    _quad                     ( nullptr ),
    _transfertexture          ( nullptr ),
    _curve_geometry           ( ),
    _view                     ( original )
{
  setFocusPolicy(Qt::StrongFocus);
}


///////////////////////////////////////////////////////////////////////
glwidget::~glwidget()
{
  if ( _partition_program         ) delete _partition_program;

  if ( _db_program   ) delete _db_program   ;
  if ( _db_trimdata  ) delete _db_trimdata  ;
  if ( _db_celldata  ) delete _db_celldata  ;
  if ( _db_curvelist ) delete _db_curvelist ;
  if ( _db_curvedata ) delete _db_curvedata ;

  if ( _cmb_program )    delete _cmb_program ;
  if ( _cmb_partition )  delete _cmb_partition ;
  if ( _cmb_contourlist ) delete _cmb_contourlist;
  if ( _cmb_curvelist   ) delete _cmb_curvelist  ;
  if ( _cmb_curvedata   ) delete _cmb_curvedata  ;
  if ( _cmb_pointdata   ) delete _cmb_pointdata  ;

  if ( _kd_program )     delete _kd_program ;
  if ( _kd_partition )   delete _kd_partition ;
  if ( _kd_contourlist ) delete _kd_contourlist;
  if ( _kd_curvelist   ) delete _kd_curvelist  ;
  if ( _kd_curvedata   ) delete _kd_curvedata  ;
  if ( _kd_pointdata   ) delete _kd_pointdata  ;

  if ( _quad                      ) delete _quad                       ;
  if ( _transfertexture           ) delete _transfertexture            ;
}


///////////////////////////////////////////////////////////////////////
void glwidget::initializeGL()
{}


///////////////////////////////////////////////////////////////////////
void                    
glwidget::open ( std::list<std::string> const& files )
{
  struct {
    std::unordered_map<gpucast::trimdomain::curve_ptr, gpucast::trimdomain_serializer::address_type>          curve_map;
    std::unordered_map<gpucast::beziersurface::trimdomain_ptr, gpucast::trimdomain_serializer::address_type>  domain_map;
    std::vector<gpucast::gl::vec4f> part;
    std::vector<gpucast::gl::vec4f> cells;
    std::vector<gpucast::gl::vec4f> curvelist;
    std::vector<gpucast::gl::vec3f> curvedata;
  } classic_partition;

  struct {
    std::unordered_map<gpucast::trimdomain::curve_ptr, gpucast::trimdomain_serializer::address_type>           curve_map;
    std::unordered_map<gpucast::beziersurface::trimdomain_ptr, gpucast::trimdomain_serializer::address_type>   domain_map;
    std::unordered_map<gpucast::trimdomain::contour_segment_ptr, gpucast::trimdomain_serializer::address_type> segment_map;
    std::vector<gpucast::gl::vec4f> part;
    std::vector<gpucast::gl::vec2f> contourlist;
    std::vector<gpucast::gl::vec4f> curvelist;
    std::vector<float>       curvedata;
    std::vector<gpucast::gl::vec3f> pointdata;
  } contour_partition;

  struct {
    std::unordered_map<gpucast::trimdomain::curve_ptr, gpucast::trimdomain_serializer::address_type>           curve_map;
    std::unordered_map<gpucast::beziersurface::trimdomain_ptr, gpucast::trimdomain_serializer::address_type>   domain_map;
    std::unordered_map<gpucast::trimdomain::contour_segment_ptr, gpucast::trimdomain_serializer::address_type> segment_map;
    std::vector<gpucast::gl::vec4f> part;
    std::vector<gpucast::gl::vec2f> contourlist;
    std::vector<gpucast::gl::vec4f> curvelist;
    std::vector<float>       curvedata;
    std::vector<gpucast::gl::vec3f> pointdata;
  } contour_kd_partition;

  gpucast::trimdomain_serializer_double_binary      db_serializer;
  gpucast::trimdomain_serializer_contour_map_kd     kd_serializer;
  gpucast::trimdomain_serializer_contour_map_binary cm_serializer;

  for ( std::string const& filename : files )
  {
    gpucast::igs_loader igs_loader; 
    std::shared_ptr<gpucast::nurbssurfaceobject>  nobj = igs_loader.load(filename);
    std::shared_ptr<gpucast::beziersurfaceobject> bobj = std::make_shared<gpucast::beziersurfaceobject>();

    gpucast::surface_converter converter;
    converter.convert(nobj, bobj);
    //bobj->init();

    _objects[filename] = bobj;

    // initial trim generation
    for ( auto i = bobj->begin(); i != bobj->end(); ++i )
    {
      db_serializer.serialize ( (**i).domain(), 
                                classic_partition.domain_map, 
                                classic_partition.curve_map, 
                                classic_partition.part, 
                                classic_partition.cells, 
                                classic_partition.curvelist, 
                                classic_partition.curvedata );

      cm_serializer.serialize ( (**i).domain(),
                                contour_partition.domain_map, 
                                contour_partition.curve_map, 
                                contour_partition.segment_map, 
                                contour_partition.part, 
                                contour_partition.contourlist, 
                                contour_partition.curvelist, 
                                contour_partition.curvedata, 
                                contour_partition.pointdata );

      kd_serializer.serialize ( (**i).domain(),
                                contour_kd_partition.domain_map, 
                                contour_kd_partition.curve_map, 
                                contour_kd_partition.segment_map, 
                                contour_kd_partition.part, 
                                contour_kd_partition.contourlist, 
                                contour_kd_partition.curvelist, 
                                contour_kd_partition.curvedata, 
                                contour_kd_partition.pointdata );
    }
  }

  std::cout << "classic :" << std::endl;
  std::cout << " vpartition  : " << classic_partition.part.size() *      sizeof(gpucast::gl::vec4f) << " Bytes " << std::endl;
  std::cout << " cell data   : " << classic_partition.cells.size() *     sizeof(gpucast::gl::vec4f) << " Bytes " << std::endl;
  std::cout << " curve lists : " << classic_partition.curvelist.size() * sizeof(gpucast::gl::vec4f) << " Bytes " << std::endl;
  std::cout << " curve data  : " << classic_partition.curvedata.size() * sizeof(gpucast::gl::vec3f) << " Bytes " << std::endl;
  std::cout << " total       : " << classic_partition.part.size() *      sizeof(gpucast::gl::vec4f) +
                                    classic_partition.cells.size() *     sizeof(gpucast::gl::vec4f) + 
                                    classic_partition.curvelist.size() * sizeof(gpucast::gl::vec4f) + 
                                    classic_partition.curvedata.size() * sizeof(gpucast::gl::vec3f)
                                    << " Bytes " << std::endl;

  std::cout << "contour map :" << std::endl;
  std::cout << " part         : " << contour_partition.part.size() *         sizeof(gpucast::gl::vec4f) << " Bytes " << std::endl;
  std::cout << " contour lists : " << contour_partition.contourlist.size() * sizeof(gpucast::gl::vec2f) << " Bytes " << std::endl;
  std::cout << " curve lists   : " << contour_partition.curvelist.size() *   sizeof(gpucast::gl::vec4f) << " Bytes " << std::endl;
  std::cout << " curve data    : " << contour_partition.curvedata.size() *   sizeof(gpucast::gl::vec2f) << " Bytes " << std::endl;
  std::cout << " point data    : " << contour_partition.pointdata.size() *   sizeof(gpucast::gl::vec3f) << " Bytes " << std::endl;
  std::cout << " total       : "   << contour_partition.part.size() *        sizeof(gpucast::gl::vec4f) +
                                      contour_partition.contourlist.size() * sizeof(gpucast::gl::vec2f) + 
                                      contour_partition.curvelist.size() *   sizeof(gpucast::gl::vec4f) + 
                                      contour_partition.curvedata.size() *   sizeof(float) + 
                                      contour_partition.pointdata.size() *   sizeof(gpucast::gl::vec3f)
                                    << " Bytes " << std::endl;

  std::cout << "contour map kd :" << std::endl;
  std::cout << " part          : " << contour_kd_partition.part.size() *        sizeof(gpucast::gl::vec4f) << " Bytes " << std::endl;
  std::cout << " contour lists : " << contour_kd_partition.contourlist.size() * sizeof(gpucast::gl::vec2f) << " Bytes " << std::endl;
  std::cout << " curve lists   : " << contour_kd_partition.curvelist.size() *   sizeof(gpucast::gl::vec4f) << " Bytes " << std::endl;
  std::cout << " curve data    : " << contour_kd_partition.curvedata.size() *   sizeof(gpucast::gl::vec2f) << " Bytes " << std::endl;
  std::cout << " point data    : " << contour_kd_partition.pointdata.size() *   sizeof(gpucast::gl::vec3f) << " Bytes " << std::endl;
  std::cout << " total       : "   << contour_kd_partition.part.size() *        sizeof(gpucast::gl::vec4f) +
                                      contour_kd_partition.contourlist.size() * sizeof(gpucast::gl::vec2f) + 
                                      contour_kd_partition.curvelist.size() *   sizeof(gpucast::gl::vec4f) + 
                                      contour_kd_partition.curvedata.size() *   sizeof(float) + 
                                      contour_kd_partition.pointdata.size() *   sizeof(gpucast::gl::vec3f)
                                    << " Bytes " << std::endl;
}


///////////////////////////////////////////////////////////////////////
void                    
glwidget::clear() 
{
  // delete old polygons
  for ( gpucast::gl::line* p : _curve_geometry ) {
    delete p;
  }
  // delete invalid pointers
  _curve_geometry.clear();
}


///////////////////////////////////////////////////////////////////////
void                    
glwidget::update_view ( std::string const& name, std::size_t const index, view current_view )
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
        case contour_binary_partition :
          generate_bboxmap_view ( domain );
          break;
        case contour_binary_classification :
          serialize_contour_binary ( domain );
          break;
        case contour_map_partition :
          generate_bboxmap_view ( domain );
          break;
        case contour_map_classification :
          serialize_contour_map ( domain );
          break;
        case minification  :
          // not implemented
          break;
        default : 
          break;
      };

      _projection = gpucast::gl::ortho(float(domain->nurbsdomain().min[gpucast::trimdomain::point_type::u] ), 
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
    gpucast::gl::vec4f cpolygon_color ( 1.0f, 1.0f, 1.0f, 1.0f );
    add_gl_curve ( **curve, cpolygon_color );

    // generate bbox to draw
    gpucast::math::bbox2d bbox;
    (**curve).bbox_simple(bbox);

    gpucast::gl::vec4f bbox_color ( 1.0f, 0.0f, 0.0f, 1.0f );
    add_gl_bbox ( bbox, bbox_color );
  }
}


///////////////////////////////////////////////////////////////////////
void                    
glwidget::generate_double_binary_view ( gpucast::beziersurface::trimdomain_ptr const& domain )
{
  gpucast::trimdomain::curve_container curves = domain->curves();
  for ( auto curve = curves.begin(); curve != curves.end(); ++curve )
  {
    gpucast::gl::vec4f cpolygon_color ( 1.0f, 1.0f, 1.0f, 1.0f );
    add_gl_curve ( **curve, cpolygon_color );
  }

  gpucast::math::partition<gpucast::beziersurface::curve_point_type>  partition ( curves.begin(), curves.end() );
  partition.initialize();

  for ( auto v = partition.begin(); v != partition.end(); ++v )
  {
    gpucast::math::bbox2d bbox ( gpucast::math::point2d ( (**v).get_horizontal_interval().minimum(), (**v).get_vertical_interval().minimum() ),
                       gpucast::math::point2d ( (**v).get_horizontal_interval().maximum(), (**v).get_vertical_interval().maximum() ) );
      gpucast::gl::vec4f cell_color ( 1.0f, 1.0f, 0.0f, 1.0f );
      add_gl_bbox ( bbox, cell_color ); 
    for ( auto c = (**v).begin(); c != (**v).end(); ++c )
    {
      gpucast::math::bbox2d bbox ( gpucast::math::point2d ( (**c).get_horizontal_interval().minimum(), (**c).get_vertical_interval().minimum() ),
                         gpucast::math::point2d ( (**c).get_horizontal_interval().maximum(), (**c).get_vertical_interval().maximum() ) );
      gpucast::gl::vec4f cell_color ( 0.0f, 1.0f, 0.0f, 1.0f );
      add_gl_bbox ( bbox, cell_color ); 
    }
  }
}


///////////////////////////////////////////////////////////////////////
void                    
glwidget::generate_bboxmap_view ( gpucast::beziersurface::trimdomain_ptr const& domain )
{
  gpucast::math::contour_map_binary<double> cmap;
  for ( auto const& loop : domain->loops() )
  {
    cmap.add ( gpucast::math::contour<double> ( loop.begin(), loop.end() ) );
  }

  cmap.initialize();

  for ( gpucast::math::contour_map_binary<double>::contour_segment_ptr const& segment : cmap.monotonic_segments() )
  {
    gpucast::gl::vec4f segment_color ( 0.0f, 1.0f, 1.0f, 1.0f );
    add_gl_bbox ( segment->bbox(), segment_color );

    for ( auto c = segment->begin(); c != segment->end(); ++c ) 
    {
      gpucast::gl::vec4f curve_bbox_color ( 1.0f, 0.0f, 0.0f, 1.0f );
      gpucast::math::bbox2d bbox;
      (**c).bbox_simple(bbox);
      add_gl_bbox ( bbox, curve_bbox_color );

      gpucast::gl::vec4f cpolygon_color ( 1.0f, 1.0f, 1.0f, 1.0f );
      add_gl_curve ( **c, cpolygon_color );
    }
  }

  for ( gpucast::math::contour_map_binary<double>::contour_interval const& vslab : cmap.partition() )
  {
    for ( gpucast::math::contour_map_binary<double>::contour_cell const& cell : vslab.cells )
    {
      gpucast::gl::vec4f vslab_color ( 0.0f, 1.0f, 0.0f, 1.0f );
      gpucast::math::bbox2d vbox ( gpucast::math::point2d ( cell.interval_u.minimum(), cell.interval_v.minimum()), gpucast::math::point2d ( cell.interval_u.maximum(), cell.interval_v.maximum()));
       add_gl_bbox ( vbox, vslab_color );
      for ( gpucast::math::contour_map_binary<double>::contour_segment_ptr const& contour : cell.overlapping_segments )
      {
        gpucast::gl::vec4f cell_color ( 0.0f, 1.0f, 0.0f, 1.0f );
        gpucast::math::bbox2d ubox = contour->bbox();
        add_gl_bbox ( ubox, cell_color );
      }
    }
  }
}


///////////////////////////////////////////////////////////////////////
void                    
glwidget::serialize_double_binary ( gpucast::beziersurface::trimdomain_ptr const& domain )
{
  if ( !_db_trimdata )   _db_trimdata  = new gpucast::gl::texturebuffer;
  if ( !_db_celldata )   _db_celldata  = new gpucast::gl::texturebuffer;
  if ( !_db_curvelist )  _db_curvelist = new gpucast::gl::texturebuffer;
  if ( !_db_curvedata )  _db_curvedata = new gpucast::gl::texturebuffer;

  std::unordered_map<gpucast::trimdomain::curve_ptr, gpucast::trimdomain_serializer::address_type>          referenced_curves;
  std::unordered_map<gpucast::beziersurface::trimdomain_ptr, gpucast::trimdomain_serializer::address_type>  referenced_domains;

  std::vector<gpucast::gl::vec4f> trimdata(1);
  std::vector<gpucast::gl::vec4f> celldata(1);
  std::vector<gpucast::gl::vec4f> curvelists(1);
  std::vector<gpucast::gl::vec3f> curvedata(1);

  gpucast::trimdomain_serializer_double_binary serializer;
  _trimid = serializer.serialize ( domain, referenced_domains, referenced_curves, trimdata, celldata, curvelists, curvedata ) ;

  std::size_t size_bytes = ( ( trimdata.size() - 1 )   * sizeof(gpucast::gl::vec4f) +
                             ( celldata.size() - 1 )   * sizeof(gpucast::gl::vec4f) + 
                             ( curvelists.size() - 1 ) * sizeof(gpucast::gl::vec4f) + 
                             ( curvedata.size() - 1 )  * sizeof(gpucast::gl::vec3f) );

  mainwindow* win = dynamic_cast<mainwindow*>(parent());
  if ( win )
  {
    win->show_memusage(size_bytes);
  }

  _db_trimdata->update  ( trimdata.begin(), trimdata.end() );
  _db_celldata->update  ( celldata.begin(), celldata.end() );
  _db_curvelist->update ( curvelists.begin(), curvelists.end() );
  _db_curvedata->update ( curvedata.begin(), curvedata.end() );
                       
  _db_trimdata->format  ( GL_RGBA32F );
  _db_celldata->format  ( GL_RGBA32F );
  _db_curvelist->format ( GL_RGBA32F );
  _db_curvedata->format ( GL_RGB32F );

  _domain_size = gpucast::gl::vec2f ( domain->nurbsdomain().size() );
  _domain_min  = gpucast::gl::vec2f ( domain->nurbsdomain().min );
}


void                    
glwidget::serialize_contour_binary ( gpucast::beziersurface::trimdomain_ptr const& domain )
{
  if ( !_cmb_partition )   _cmb_partition  = new gpucast::gl::texturebuffer;
  if ( !_cmb_contourlist ) _cmb_contourlist = new gpucast::gl::texturebuffer;
  if ( !_cmb_curvelist   ) _cmb_curvelist   = new gpucast::gl::texturebuffer;
  if ( !_cmb_curvedata   ) _cmb_curvedata   = new gpucast::gl::texturebuffer;
  if ( !_cmb_pointdata   ) _cmb_pointdata   = new gpucast::gl::texturebuffer;

  std::unordered_map<gpucast::trimdomain::curve_ptr,         gpucast::trimdomain_serializer::address_type>                                  referenced_curves;
  std::unordered_map<gpucast::beziersurface::trimdomain_ptr, gpucast::trimdomain_serializer::address_type>                                  referenced_domains;
  std::unordered_map<gpucast::trimdomain_serializer_contour_map_binary::contour_segment_ptr, gpucast::trimdomain_serializer::address_type>  referenced_contours;

  std::vector<gpucast::gl::vec4f> partition(1);
  std::vector<gpucast::gl::vec2f> contourlist(1);
  std::vector<gpucast::gl::vec4f> curvelist(1);
  std::vector<float> curvedata(1);
  std::vector<gpucast::gl::vec3f> pointdata(1);

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

  std::size_t size_bytes = ( (partition.size()  - 1) * sizeof(gpucast::gl::vec4f) + 
                             (contourlist.size()- 1) * sizeof(gpucast::gl::vec2f) + 
                             (curvelist.size()  - 1) * sizeof(gpucast::gl::vec4f) + 
                             (curvedata.size()  - 1) * sizeof(float) + 
                             (pointdata.size()  - 1) * sizeof(gpucast::gl::vec3f) );

  mainwindow* win = dynamic_cast<mainwindow*>(parent());
  if ( win )
  {
    win->show_memusage(size_bytes);
  }

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

  _domain_size = gpucast::gl::vec2f ( domain->nurbsdomain().size() );
  _domain_min  = gpucast::gl::vec2f ( domain->nurbsdomain().min );
}

///////////////////////////////////////////////////////////////////////
void                    
glwidget::serialize_contour_map ( gpucast::beziersurface::trimdomain_ptr const& domain )
{}

///////////////////////////////////////////////////////////////////////
void                    
glwidget::add_gl_curve ( gpucast::beziersurface::curve_type const& curve, gpucast::gl::vec4f const& color )
{
  // draw curve using a fixed number of sample points
  unsigned const samples = 50;
  std::vector<gpucast::gl::vec4f> curve_points;
  for ( unsigned sample = 0; sample <= samples; ++sample ) 
  {
    gpucast::math::point2d p;
    curve.evaluate(float(sample)/samples, p);
    curve_points.push_back( gpucast::gl::vec4f ( p[0], p[1], -1.0f, 1.0f ) );
  }

  std::vector<gpucast::gl::vec4f> curve_colors ( curve_points.size(), color );
  gpucast::gl::line* gl_curve = new gpucast::gl::line ( curve_points, 0, 1, 2);
  gl_curve->set_color ( curve_colors );
  _curve_geometry.push_back ( gl_curve ); 

}


///////////////////////////////////////////////////////////////////////
void                    
glwidget::add_gl_bbox (  gpucast::math::bbox2d const& bbox, gpucast::gl::vec4f const& color )
{
  std::vector<gpucast::gl::vec4f> bbox_points;

  bbox_points.push_back ( gpucast::gl::vec4f ( float(bbox.min[0]), float(bbox.min[1]), -2.0f, 1.0f ) );
  bbox_points.push_back ( gpucast::gl::vec4f ( float(bbox.max[0]), float(bbox.min[1]), -2.0f, 1.0f ) );
  bbox_points.push_back ( gpucast::gl::vec4f ( float(bbox.max[0]), float(bbox.max[1]), -2.0f, 1.0f ) );
  bbox_points.push_back ( gpucast::gl::vec4f ( float(bbox.min[0]), float(bbox.max[1]), -2.0f, 1.0f ) );
  bbox_points.push_back ( gpucast::gl::vec4f ( float(bbox.min[0]), float(bbox.min[1]), -2.0f, 1.0f ) );
  bbox_points.push_back ( gpucast::gl::vec4f ( float(bbox.max[0]), float(bbox.max[1]), -2.0f, 1.0f ) );
  bbox_points.push_back ( gpucast::gl::vec4f ( float(bbox.max[0]), float(bbox.min[1]), -2.0f, 1.0f ) );
  bbox_points.push_back ( gpucast::gl::vec4f ( float(bbox.min[0]), float(bbox.max[1]), -2.0f, 1.0f ) );

  std::vector<gpucast::gl::vec4f> bbox_colors ( bbox_points.size(), color );

  gpucast::gl::line* gl_bbox = new gpucast::gl::line ( bbox_points, 0, 1, 2);
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
    case original :
    case double_binary_partition :
    case contour_map_partition :
    case contour_binary_partition :
      {
        _partition_program->begin();
        _partition_program->set_uniform_matrix4fv("mvp", 1, false, &_projection[0]); 
        std::for_each ( _curve_geometry.begin(), _curve_geometry.end(), std::bind(&gpucast::gl::line::draw, std::placeholders::_1, 1.0f));
        _partition_program->end();
        break;
      }
    case double_binary_classification :
      {
        _db_program->begin();
        {
          _db_program->set_uniform1i ( "trimid", _trimid );
          _db_program->set_uniform1i ( "width",  _width );
          _db_program->set_uniform1i ( "height", _height );
                                                  
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
    case contour_binary_classification :
            
        _cmb_program->begin();
        {
          _cmb_program->set_uniform1i ( "trimid", _trimid );
          _cmb_program->set_uniform1i ( "width",  _width );
          _cmb_program->set_uniform1i ( "height", _height );
                                                  
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
      break;
    case contour_map_classification :
      
      break;
    case minification  :
      // not implemented
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
    _quad = new gpucast::gl::plane(0, -1, -1);
  }

  if ( !_transfertexture )
  {
    _transfertexture = new gpucast::gl::texture1d;
    gpucast::gl::transferfunction<gpucast::gl::vec4f> tf;
    tf.set ( 0, gpucast::gl::vec4f(0.0f, 0.0f, 0.5f, 1.0f) );
    tf.set ( 60, gpucast::gl::vec4f(0.0f, 0.7f, 0.5f, 1.0f) );
    tf.set ( 120, gpucast::gl::vec4f(0.0f, 0.9f, 0.0f, 1.0f) );
    tf.set ( 180, gpucast::gl::vec4f(0.9f, 0.9f, 0.0f, 1.0f) );
    tf.set ( 255, gpucast::gl::vec4f(0.9f, 0.0f, 0.0f, 1.0f) );

    std::vector<gpucast::gl::vec4f> samples;
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
  if ( _partition_program ) delete _partition_program;
  if ( _db_program )        delete _db_program;
  if ( _cmb_program )       delete _cmb_program;
  if ( _kd_program )        delete _kd_program;

  _partition_program  = _init_program ( "../shader/partition.vert.glsl", "../shader/partition.frag.glsl" );
  _db_program         = _init_program ( "../shader/domain.vert.glsl", "../shader/double_binary.frag.glsl" );
  _cmb_program        = _init_program ( "../shader/domain.vert.glsl", "../shader/contour_binary.frag.glsl" );
  _kd_program         = _init_program ( "../shader/domain.vert.glsl", "../shader/contour.frag.glsl" );
}


/////////////////////////////////////////////////////////////////////////////
gpucast::gl::program*
glwidget::_init_program ( std::string const& vertexshader_filename,
                          std::string const& fragmentshader_filename )
{
  try {
    gpucast::gl::vertexshader     vs;
    gpucast::gl::fragmentshader   fs;

    gpucast::gl::program* p = new gpucast::gl::program;

    if ( boost::filesystem::exists(vertexshader_filename) )
    {
      vs.load(vertexshader_filename);
      vs.compile();
      if ( !vs.log().empty() ) {
        std::cout << vertexshader_filename << " log : " << vs.log() << std::endl;
      }
      p->add(&vs);
    } else {
      throw std::runtime_error("renderer::_init_shader (): Couldn't open file " + vertexshader_filename);
    }

    if ( boost::filesystem::exists(fragmentshader_filename)  )
    {
      fs.load(fragmentshader_filename);
      fs.compile();
 
      if ( !fs.log().empty() ) {
        std::cout << fragmentshader_filename << " log : " << fs.log() << std::endl;
      }
      p->add(&fs);
    } else {
      throw std::runtime_error("renderer::_init_shader (): Couldn't open file " + fragmentshader_filename);
    }

    // link all shaders
    p->link();

    if ( !p->log().empty() )
    {
      // stream log to std output
      std::cout << " program log : " << p->log() << std::endl;
    }

    return p;
  } catch ( std::exception& e ) {
    std::cerr << "renderer::init_program(): failed to init program : " << vertexshader_filename << ", " << fragmentshader_filename << "( " << e.what () << ")\n";
    return nullptr;
  }
}

