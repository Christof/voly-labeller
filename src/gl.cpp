#include "./gl.h"
#include <QDebug>
#include <QOpenGLPaintDevice>

Gl::Gl()
{
}

Gl::~Gl()
{
  if (paintDevice)
  {
    delete paintDevice;
    paintDevice = nullptr;
  }
}

void Gl::initialize(QSize size)
{
  qDebug() << "Initialize OpenGL";
  initializeOpenGLFunctions();
  paintDevice = new QOpenGLPaintDevice();
  setSize(size);
}

void Gl::setSize(QSize size)
{
  this->size = size;
  if (paintDevice)
    paintDevice->setSize(size);
}
