/****************************************************************************
**
** Copyright (C) 2016 The Qt Company Ltd.
** Contact: https://www.qt.io/licensing/
**
** This file is part of the examples of the Qt Toolkit.
**
** $QT_BEGIN_LICENSE:BSD$
** Commercial License Usage
** Licensees holding valid commercial Qt licenses may use this file in
** accordance with the commercial license agreement provided with the
** Software or, alternatively, in accordance with the terms contained in
** a written agreement between you and The Qt Company. For licensing terms
** and conditions see https://www.qt.io/terms-conditions. For further
** information use the contact form at https://www.qt.io/contact-us.
**
** BSD License Usage
** Alternatively, you may use this file under the terms of the BSD license
** as follows:
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
**   * Neither the name of The Qt Company Ltd nor the names of its
**     contributors may be used to endorse or promote products derived
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

#include <QtWidgets>
#include <iostream>

#include "slidergroup.hpp"

//////////////////////////////////////////////////////////////////////////////////
SlidersGroup::SlidersGroup(Qt::Orientation orientation, 
                           const QString &title,
                           int minimum,
                           int maximum,
                           QWidget *parent)
  : QGroupBox(title, parent)
{
  _slider = new QSlider(orientation);
  _slider->setFocusPolicy(Qt::StrongFocus);
  _slider->setTickPosition(QSlider::TicksBothSides);
  _slider->setTickInterval(10);
  _slider->setSingleStep(1);

  connect(_slider, &QSlider::valueChanged, this, &SlidersGroup::valueChanged);
  connect(_slider, &QSlider::valueChanged, this, &SlidersGroup::updateLabels);

  QBoxLayout *slidersLayout = new QBoxLayout(orientation == Qt::Horizontal ? QBoxLayout::LeftToRight : QBoxLayout::TopToBottom);
  slidersLayout->addWidget(_slider);
  
  _label_current = new QLabel(tr(std::to_string(minimum).c_str()));
  slidersLayout->addWidget(_label_current);

  setLayout(slidersLayout);

  slidersLayout->setAlignment(_label_current, Qt::AlignCenter);

  setMinimum(minimum);
  setMaximum(maximum);
}

//////////////////////////////////////////////////////////////////////////////////
void SlidersGroup::setValue(int value)
{
  _slider->setValue(value);
  updateLabels();
}

//////////////////////////////////////////////////////////////////////////////////
void SlidersGroup::setMinimum(int value)
{
  _min = value;
  _slider->setMinimum(value);
  updateLabels();
}

//////////////////////////////////////////////////////////////////////////////////
void SlidersGroup::setMaximum(int value)
{
  _max = value;
  _slider->setMaximum(value);
  updateLabels();
}

//////////////////////////////////////////////////////////////////////////////////
void SlidersGroup::invertAppearance(bool invert)
{
  _slider->setInvertedAppearance(invert);
}

//////////////////////////////////////////////////////////////////////////////////
void SlidersGroup::invertKeyBindings(bool invert)
{
  _slider->setInvertedControls(invert);
}

//////////////////////////////////////////////////////////////////////////////////
void SlidersGroup::updateLabels()
{
  _label_current->setText(std::to_string(_slider->sliderPosition()).c_str());
}


//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
QFloatSlider::QFloatSlider(Qt::Orientation orientation, float minimum, float maximum, QWidget *parent)
  : QSlider(orientation, parent),
  _minimum(minimum),
  _maximum(maximum)
{
  connect(this, SIGNAL(valueChanged(int)), this, SLOT(notifyValueChanged(int)));

  setMinimum(0);
  setMaximum(QFLOATSLIDER_MAX_STEPS);
}

//////////////////////////////////////////////////////////////////////////////////
void QFloatSlider::notifyValueChanged(int value) {
  float percentage = float(value) / QFLOATSLIDER_MAX_STEPS;
  emit floatValueChanged(_minimum + percentage * (_maximum - _minimum));
}

//////////////////////////////////////////////////////////////////////////////////
float QFloatSlider::getfloatValue() const {
  float percentage = float(sliderPosition()) / QFLOATSLIDER_MAX_STEPS;
  return _minimum + percentage * (_maximum - _minimum);
}

//////////////////////////////////////////////////////////////////////////////////
void QFloatSlider::setfloatValue(float f) {
  setValue(int(QFLOATSLIDER_MAX_STEPS * (f - _minimum) / (_maximum - _minimum)));
}

//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
FloatSlidersGroup::FloatSlidersGroup(Qt::Orientation orientation,
                                     const QString &title,
                                     float minimum,
                                     float maximum,
                                     QWidget *parent)
  : QGroupBox(title, parent),
  _min(minimum),
  _max(maximum)
{

  _slider = new QFloatSlider(orientation, minimum, maximum, this);
  _slider->setFocusPolicy(Qt::StrongFocus);
  _slider->setTickPosition(QSlider::TicksBothSides);

  connect(_slider, &QFloatSlider::floatValueChanged, this, &FloatSlidersGroup::valueChanged);
  connect(_slider, &QFloatSlider::valueChanged, this, &FloatSlidersGroup::updateLabels);

  QBoxLayout *slidersLayout = new QBoxLayout(orientation == Qt::Horizontal ? QBoxLayout::LeftToRight : QBoxLayout::TopToBottom);
  slidersLayout->addWidget(_slider);
  
  _label_current = new QLabel(tr(std::to_string(_min).c_str()));

  slidersLayout->addWidget(_label_current);
  slidersLayout->setAlignment(_label_current, Qt::AlignCenter);

  setLayout(slidersLayout);
}

//////////////////////////////////////////////////////////////////////////////////
void FloatSlidersGroup::invertAppearance(bool invert)
{
  _slider->setInvertedAppearance(invert);
}

//////////////////////////////////////////////////////////////////////////////////
void FloatSlidersGroup::invertKeyBindings(bool invert)
{
  _slider->setInvertedControls(invert);
}

//////////////////////////////////////////////////////////////////////////////////
void FloatSlidersGroup::setValue(float f)
{
  _slider->setfloatValue(f);
}

//////////////////////////////////////////////////////////////////////////////////
void FloatSlidersGroup::updateLabels()
{
  _label_current->setText(std::to_string(_slider->getfloatValue()).c_str());
}