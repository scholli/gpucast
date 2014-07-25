
#include <iostream>
#include <map>
#include <memory>

#include <GL/glew.h>
#include <gpucast/gl/glut/window.hpp>
#include <gpucast/gl/util/init_glew.hpp>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include <gpucast/volume/uid.hpp>
#include <gpucast/volume/volume_converter.hpp>
#include <gpucast/volume/volume_renderer.hpp>
#include <gpucast/volume/beziervolumeobject.hpp>
#include <gpucast/volume/isosurface/fragment/isosurface_renderer_interval_sampling.hpp>
#include <gpucast/volume/isosurface/octree/isosurface_renderer_octreebased.hpp>
#include <gpucast/volume/isosurface/splat/isosurface_renderer_splatbased.hpp>
#include <gpucast/volume/import/mesh3d.hpp>
#include <gpucast/volume/import/xml.hpp>
#include <gpucast/volume/import/xml2.hpp>

// namespace alias
namespace bf = boost::filesystem;
namespace bp = boost::program_options;

// valid file extensions
enum filetype  { xml, mesh3d, count };

// after checks : open and serialize
void convert ( int argc, char* argv[],
               filetype type,
               bf::path const& inputfile,
               bf::path const& outputfile,
               bool preprocess,
               std::string const& renderer_name,
               bool crop,
               bool normalize_attributes,
               std::string const& crop_name,
               bool displace,
               std::string const& displace_attrib,
               bool paralleliped,
               bool greedy_binning )
{
  std::shared_ptr<gpucast::nurbsvolumeobject> nurbsobject ( new gpucast::nurbsvolumeobject );
  
  try {
    switch (type) 
    {
    case xml : 
      {
        //gpucast::xml_loader xml_importer;
        gpucast::xml2_loader xml_importer;
        xml_importer.load ( inputfile.string(), nurbsobject, displace, displace_attrib );
      } break;
    case mesh3d : 
      {
        gpucast::mesh3d_loader mesh_importer;
        mesh_importer.load ( inputfile.string(), nurbsobject, displace, displace_attrib );          
      } break;
    } 
  } catch ( std::exception& e) {
    std::cerr << "Loading " << inputfile.string() << " failed." << e.what() << std::endl;
  }

  if ( crop ) 
  {
    nurbsobject->crop(crop_name);
  }

  if ( normalize_attributes )
  {
    nurbsobject->normalize_attribute ();
  }

  // convert loaded object to bezierobject
  std::shared_ptr<gpucast::beziervolumeobject> bezierobject (new gpucast::beziervolumeobject);
  gpucast::volume_converter converter;
  converter.convert(nurbsobject, bezierobject);

  std::fstream ofstr ( outputfile.string().c_str(), std::ios::out | std::ios::binary );
  bezierobject->write(ofstr);
  ofstr.close();
  std::cerr << "Saving " << outputfile.string() << " succeed." << std::endl;

  if ( preprocess )
  {
    try {
      std::map<std::string, gpucast::volume_renderer*> renderer;
      renderer["octree"]          = new gpucast::isosurface_renderer_octreebased(argc, argv);
      renderer["face_interval"]   = new gpucast::isosurface_renderer_interval_sampling(argc, argv);
      renderer["splatting"]       = new gpucast::isosurface_renderer_splatbased(argc, argv);
      //renderer["convex_hull"]     = new gpucast::isosurface_renderer_convex_hull;

      if ( !renderer.count(renderer_name) ) {
        throw std::runtime_error("Renderer invalid. Possiple renderers are: octree, face_interval, splatting");
      }

      renderer[renderer_name]->set_attributebounds ( gpucast::volume_renderer::attribute_interval ( nurbsobject->bbox(crop_name).min[0], nurbsobject->bbox(crop_name).max[0] ));

      std::string binary_extension;
      if ( renderer_name == "octree" )          binary_extension = ".ocb";
      if ( renderer_name == "face_interval" )   binary_extension = ".fib";
      if ( renderer_name == "splatting" )       binary_extension = ".spb";

      boost::filesystem::path binary_name = (outputfile.branch_path() / boost::filesystem::basename(outputfile)).string() + crop_name + binary_extension;

      std::fstream renderinfo_ostr;
      renderinfo_ostr.open (binary_name.c_str() , std::ios::out | std::ios::binary );
      if ( paralleliped ) 
      {
        if ( greedy_binning ) {
          renderer[renderer_name]->init(bezierobject, crop_name); 
        } else {
          renderer[renderer_name]->init(bezierobject, crop_name); 
        }
      } else {
        if ( greedy_binning ) {
          renderer[renderer_name]->init(bezierobject, crop_name); 
        } else {
          renderer[renderer_name]->init(bezierobject, crop_name); 
        }
      }
      renderer[renderer_name]->write(renderinfo_ostr);
      renderinfo_ostr.close();
      std::cerr << "Saving " << binary_name << " succeed." << std::endl;

    } catch ( std::exception& e ) {
      std::cerr << "Saving " << outputfile.string() << " failed." << e.what() << std::endl;
    }
  }

}

void init_file_extension_map ( std::map<std::string, filetype>& map )
{
  map.clear();

  map[".xml"] = xml;
  map[".dat"] = mesh3d;
}


int main(int argc, char* argv[])
{
  gpucast::gl::glutwindow::init(argc, argv, 100, 100, 10, 10, 4, 1, true);

  gpucast::gl::glutwindow::instance().printinfo(std::cout);

  gpucast::gl::init_glew(std::cout);

  // initialize valid file type map
  std::map<std::string, filetype> valid_filetypes;
  init_file_extension_map ( valid_filetypes );

  try { 
    bp::options_description options ("Converter Options");
    options.add_options()
      ("help", "Usage: volume_converter -i [input] -o [output]")
      ("input,i", bp::value<std::string>(), "Input file")
      ("output,o", bp::value<std::string>(), "Ouput file")
      ("preprocess,p", bp::value<std::string>(), "Save preprocessing result. Available renderer: octree, face_interval, unified_sampler, splatting, convex_hull")
      ("displace,d", bp::value<std::string>(), "Displace mesh")
      ("crop,c", bp::value<std::string>(), "Extract single information")
      ("paralleliped", "Use parallelipeds as proxy volume")
      ("greedy,g", "Use greedy binning algorithm")
      ("normalize,n", "Normalize attributes")
      ;

    bp::variables_map arguments;
    bp::store( bp::parse_command_line(argc, argv, options), arguments );

    if ( arguments.count("help") ) {
      std::cout << options.find("help", true).description() << std::endl;
      return 0;
    }

    if ( arguments.count("input") && arguments.count("output") ) 
    {
      bf::path inputfile  ( arguments.at ("input").as<std::string>() );
      bf::path outputfile ( arguments.at ("output").as<std::string>() );

      bool preprocess           = arguments.count("preprocess") > 0;
      std::string renderer_name;
      if ( preprocess ) 
      {
        renderer_name = arguments.at ("preprocess").as<std::string>();
      }

      bool displace             = arguments.count("displace") > 0;
      bool crop                 = arguments.count("crop") > 0;
      bool paralleliped         = arguments.count("paralleliped") > 0;
      bool greedy_binning       = arguments.count("greedy") > 0;
      bool normalize            = arguments.count("normalize") > 0;

      std::string cropped_attribute;
      std::string displace_attribute;
      
      if ( crop )
      {
        cropped_attribute = arguments.at ("crop").as<std::string>();
      }

      if ( displace )
      {
        displace_attribute = arguments.at ("displace").as<std::string>();
      }

      if ( bf::exists(inputfile) ) 
      {
        std::string file_extension = bf::extension ( inputfile );

        if ( valid_filetypes.count(file_extension) ) 
        {
          // finally, convert file
          convert ( argc, argv, valid_filetypes.at(file_extension), inputfile, outputfile, preprocess, renderer_name, crop, normalize, cropped_attribute, displace, displace_attribute, paralleliped, greedy_binning );
        } else {
          throw std::runtime_error("Cannot handle input file type.");
        }
      } else {
        throw std::runtime_error("Cannot find input file.");
      }
    } else {
      throw std::runtime_error("Wrong number of arguments. Use --help to see necessary arguments.");
    }

  } catch ( std::exception& e ) {
    std::cerr << "Conversion failed..." << e.what() << std::endl;
  }

  return 0;
}

