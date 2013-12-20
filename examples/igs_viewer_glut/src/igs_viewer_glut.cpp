/********************************************************************************
* 
* Copyright (C) 2007-2010 Bauhaus-Universitaet Weimar                                               
*
*********************************************************************************
*
*  module     : glut_test.cpp                                        
*  project    : gpucast 
*  description: 
*
********************************************************************************/

// header, system
#include <iostream>
#include <exception>

// header, project
#include <application.hpp>

#include <vector>
#include <iterator>


///////////////////////////////////////////////////////////////////////////////
int 
main(int argc, char** argv)
{
  application::init(argc, argv);
   
  try {
    application::instance().run();
  } 
  catch (std::exception s) {
    std::cout << s.what() << std::endl;
    return 1;
  }
   
  return 0;
}
