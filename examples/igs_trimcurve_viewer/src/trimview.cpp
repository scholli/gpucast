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
#include <GL/glew.h>
#include <QtWidgets/QApplication>

#include <mainwindow.hpp>
#include <gpucast/gl/error.hpp>

class volume_viewer : public QApplication
{
public :
  volume_viewer ( int argc, char** argv ) : QApplication(argc, argv ) {}
  virtual ~volume_viewer() {}
#if 1
  /* virtual */ bool notify ( QObject * receiver, QEvent * e )
  {
    bool result = false;
    try {
      result = QApplication::notify(receiver, e);
    } catch  ( std::exception& e ) {
      std::cerr << e.what() << std::endl;
    }
    return result;
  }
#endif
};


int main(int argc, char **argv)
{
  try 
  {
    volume_viewer app(argc, argv);
    if ( argc == 3 ) 
    {
      mainwindow win(argc, argv, std::atoi(argv[1]), std::atoi(argv[2]));
      app.exec();
    } else {
      mainwindow win(argc, argv, 1600, 1200);
      app.exec();
    }
  } 
  catch ( std::exception& e ) 
  {
    std::cerr << e.what() << std::endl;
    system("pause");
    return 0;
  } 
}
