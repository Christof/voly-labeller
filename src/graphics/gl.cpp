#include "./gl.h"
#include <QDebug>
#include <QOpenGLPaintDevice>

namespace Graphics
{

QLoggingCategory openGlChan("OpenGl");

Gl::Gl()
{
}

Gl::~Gl()
{
  qCInfo(openGlChan) << "Destructor of Gl";
  if (paintDevice)
  {
    delete paintDevice;
    paintDevice = nullptr;
  }
}

void Gl::initialize(QOpenGLContext *context, QSize size)
{
  qCInfo(openGlChan) << "Initialize OpenGL";
  initializeOpenGLFunctions();
  glCheckError();

  bool hasShaderBufferLoad = context->hasExtension("GL_NV_shader_buffer_load");
  qCWarning(openGlChan) << "Has GL_NV_shader_buffer_load:"
                        << hasShaderBufferLoad;
  shaderBufferLoad = new QOpenGLExtension_NV_shader_buffer_load();
  shaderBufferLoad->initializeOpenGLFunctions();
  glCheckError();

  bool hasBindlessTexture = context->hasExtension("GL_NV_bindless_texture");
  qCWarning(openGlChan) << "Has GL_NV_bindless_texture:" << hasBindlessTexture;
  bindlessTexture = new QOpenGLExtension_NV_bindless_texture();
  bindlessTexture->initializeOpenGLFunctions();
  glCheckError();

  qCWarning(openGlChan) << "Has GL_ARB_sparse_texture:"
                        << context->hasExtension("GL_ARB_sparse_texture");

  glTexturePageCommitmentEXT = reinterpret_cast<TexturePageCommitmentEXT>(
      context->getProcAddress("glTexturePageCommitmentEXT"));

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

QOpenGLExtension_NV_bindless_texture *Gl::getBindlessTexture() const
{
  return bindlessTexture;
}

}  // namespace Graphics
