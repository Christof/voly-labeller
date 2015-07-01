#include "./gl.h"
#include <QDebug>
#include <QOpenGLPaintDevice>

namespace Graphics
{

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

  bool hasDirectStateAccess =
      context->hasExtension("GL_EXT_direct_state_access");
  qWarning() << "Has GL_EXT_direct_state_access:" << hasDirectStateAccess;
  directStateAccess = new QOpenGLExtension_EXT_direct_state_access();
  directStateAccess->initializeOpenGLFunctions();
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

QOpenGLExtension_NV_shader_buffer_load *Gl::getShaderBufferLoad() const
{
  return shaderBufferLoad;
}

QOpenGLExtension_EXT_direct_state_access *Gl::getDirectStateAccess() const
{
  return directStateAccess;
}

}  // namespace Graphics
