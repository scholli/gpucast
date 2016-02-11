/********************************************************************************
*
* Copyright (C) 2014 Bauhaus University Weimar
*
*********************************************************************************
*
*  module     : helloqt5.cpp
*  project    : glpp
*  description:
*
********************************************************************************/
#pragma warning(disable: 4127) // Qt conditional expression is constant

// system includes
#include <QtWidgets/QApplication>
#include <QtWidgets/QLabel>

int main(int argc, char *argv[])
{
  QApplication app(argc, argv); 
  QLabel *label = new QLabel("Hello Qt!"); 
  label->show();
  return app.exec();

  return 0;
}