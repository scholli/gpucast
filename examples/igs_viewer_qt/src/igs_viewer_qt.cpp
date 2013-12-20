/********************************************************************************
*
* Copyright (C) 2010 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : volume_view.cpp
*  project    : glpp
*  description:
*
********************************************************************************/
#pragma warning(disable: 4127) // Qt conditional expression is constant

// system includes
#include <exception>
#include <GL/glew.h>
#include <QtGui/QApplication>

#include <mainwindow.hpp>
#include <glpp/error.hpp>

int main(int argc, char **argv)
{
  try 
  {
    QApplication app(argc, argv);
    mainwindow win(argc, argv, 1920, 1200);
    win.show();
    app.exec();
  } 
  catch ( std::exception& e ) 
  {
    std::cerr << e.what() << std::endl;
    system("pause");
    return 0;
  }
}
