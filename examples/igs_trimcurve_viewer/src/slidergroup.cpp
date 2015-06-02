/****************************************************************************
**
** Copyright (C) 2014 Digia Plc and/or its subsidiary(-ies).
** Contact: http://www.qt-project.org/legal
**
** This file is part of the examples of the Qt Toolkit.
**
** $QT_BEGIN_LICENSE:BSD$
** You may use this file under the terms of the BSD license as follows:
**
** "Redistribution and use in source and binary forms, with or without
** modification, are permitted provided that the following conditions are
** met:
**   * Redistributions of source code must retain the above copyright
**     notice, this list of conditions and the following disclaimer.
**   * Redistributions in binary form must reproduce the above copyright
**     notice, this list of conditions and the following disclaimer in
**     the documentation and/or other materials provided with the
**     distribution.
**   * Neither the name of Digia Plc and its Subsidiary(-ies) nor the names
**     of its contributors may be used to endorse or promote products derived
**     from this software without specific prior written permission.
**
**
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
** OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
** LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
** DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
** THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
** (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
** OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
**
** $QT_END_LICENSE$
**
****************************************************************************/

#include <QtGui>
#include <QtWidgets/QSlider>
#include <QtWidgets/QScrollbar>
#include <QtWidgets/QDial>
#include <QtWidgets/QLabel>
#include <QtWidgets/QBoxlayout>

#include <slidergroup.hpp>
#include <string>


SlidersGroup::SlidersGroup(Qt::Orientation orientation, const QString &title,
  QWidget *parent)
  : QGroupBox(title, parent)
{
  slider = new QSlider(orientation);
  slider->setFocusPolicy(Qt::StrongFocus);
  slider->setTickPosition(QSlider::TicksBothSides);
  slider->setTickInterval(10);
  slider->setSingleStep(1);

  label = new QLabel;
  QBoxLayout::Direction direction;

  if (orientation == Qt::Horizontal)
    direction = QBoxLayout::TopToBottom;
  else
    direction = QBoxLayout::LeftToRight;

  QBoxLayout *slidersLayout = new QBoxLayout(direction);
  slidersLayout->addWidget(slider);
  slidersLayout->addWidget(label);
  setLayout(slidersLayout);

  connect(slider, SIGNAL(valueChanged(int)), this, SIGNAL(valueChanged(int)));
  connect(slider, SIGNAL(valueChanged(int)), this, SLOT(update()));

  update();
}

void SlidersGroup::setValue(int value)
{
  slider->setValue(value);
}

void SlidersGroup::setMinimum(int value)
{
  slider->setMinimum(value);
}

void SlidersGroup::setMaximum(int value)
{
  slider->setMaximum(value);
}

void SlidersGroup::update() {
  //label->setText(tr(std::to_string(float(slider->value())/float(getMaximum() - getMinimum())).c_str() ));
  label->setText(tr(std::to_string(slider->value()).c_str()));
}

int SlidersGroup::getMinimum() const {
  return slider->minimum();
}

int SlidersGroup::getMaximum() const {
  return slider->maximum();
}

void SlidersGroup::invertAppearance(bool invert)
{
  slider->setInvertedAppearance(invert);
}

void SlidersGroup::invertKeyBindings(bool invert)
{
  slider->setInvertedControls(invert);
}