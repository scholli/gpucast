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

// system includes
#include <QtGui/QMouseEvent>
#include <iostream>




///////////////////////////////////////////////////////////////////////
glwidget::glwidget(QWidget *parent)
 : QGLWidget(parent) 
{}


///////////////////////////////////////////////////////////////////////
glwidget::~glwidget()
{}


///////////////////////////////////////////////////////////////////////
void glwidget::initializeGL()
{
 glClearColor(1.0, 0.0, 0.0, 1.0);
 glEnable(GL_DEPTH_TEST);
}


///////////////////////////////////////////////////////////////////////
void 
glwidget::resizeGL(int width, int height)
{
  int side = qMin(width, height);

  glViewport((width - side) / 2, (height - side) / 2, side, side);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();

  // todo : set projection

  glMatrixMode(GL_MODELVIEW);

  _width  = width;
  _height = height;
}


///////////////////////////////////////////////////////////////////////
void 
glwidget::paintGL()
{
  glClearColor(_intensity, 1.0, 0.0, 1.0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  _intensity += 0.0001f;

  if (_intensity > 1.0f) {
    _intensity = 0.0f;
  }

  this->update();
}


///////////////////////////////////////////////////////////////////////
void 
glwidget::mousePressEvent(QMouseEvent *event)
{
  _last_mouse_position = event->pos();
  std::cout << _last_mouse_position.y() << std::endl;
}

///////////////////////////////////////////////////////////////////////
void 
glwidget::mouseMoveEvent(QMouseEvent *event)
{
  _last_mouse_position = event->pos();
  std::cout << "motion:  " << _last_mouse_position.x() << std::endl;
}