#include "./gl.h"
#include <QDebug>
#include <QOpenGLPaintDevice>

Gl::Gl()
{
}

Gl::~Gl()
{
  qDebug() << "Destructor of Gl";
  if (paintDevice)
  {
    delete paintDevice;
    paintDevice = nullptr;
  }
}

void Gl::initialize(QOpenGLContext *context, QSize size)
{
  qDebug() << "Initialize OpenGL";
  initializeOpenGLFunctions();
  glCheckError();

  bool hasShaderBufferLoad = context->hasExtension("GL_NV_shader_buffer_load");
  qWarning() << "Has GL_NV_shader_buffer_load:" << hasShaderBufferLoad;
  shaderBufferLoad = new QOpenGLExtension_NV_shader_buffer_load();
  shaderBufferLoad->initializeOpenGLFunctions();
  glCheckError();

  paintDevice = new QOpenGLPaintDevice();
  setSize(size);
}

void Gl::setSize(QSize size)
{
  this->size = size;
  if (paintDevice)
    paintDevice->setSize(size);
}

const QOpenGLExtension_NV_shader_buffer_load *Gl::getShaderBufferLoad() const
{
  return shaderBufferLoad;
}
