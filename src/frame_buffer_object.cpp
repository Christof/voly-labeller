#include "./frame_buffer_object.h"
#include <QDebug>
#include "./gl.h"

FrameBufferObject::~FrameBufferObject()
{
  fbo->release();
  gl->glDeleteTextures(1, &depthTexture);
}

void FrameBufferObject::initialize(Gl *gl, int width, int height)
{
  bool resize = fbo.get();
  if (!resize)
  {
    this->gl = gl;
    QOpenGLFramebufferObjectFormat format;
    format.setAttachment(QOpenGLFramebufferObject::Depth);
    // format.setSamples(16);
    fbo = std::unique_ptr<QOpenGLFramebufferObject>(
        new QOpenGLFramebufferObject(width, height, format));
    qWarning() << "create fbo";
    /*
    fbo->release();
    fbo.release();
    */
    glAssert(gl->glGenTextures(1, &depthTexture));
  }

  glAssert(fbo->bind());

  qWarning() << "Resize to " << width << "x" << height;
  glAssert(gl->glBindTexture(GL_TEXTURE_2D, depthTexture));
  resizeTexture(depthTexture, width, height, GL_DEPTH_COMPONENT,
                GL_DEPTH_COMPONENT32F);
  glAssert(gl->glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                                      GL_TEXTURE_2D, depthTexture, 0));

  glAssert(gl->glBindTexture(GL_TEXTURE_2D, fbo->texture()));
  resizeTexture(fbo->texture(), width, height, GL_RGBA, GL_RGBA8);
  glAssert(gl->glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                      GL_TEXTURE_2D, fbo->texture(), 0));

  glAssert(fbo->release());
}

void FrameBufferObject::bind()
{
  // QOpenGLFramebufferObject::blitFramebuffer(fbo.get(), fbo.get());
  glAssert(fbo->bind());
}

void FrameBufferObject::unbind()
{
  glAssert(fbo->release());
}

void FrameBufferObject::bindColorTexture(unsigned int textureUnit)
{
  glAssert(gl->glActiveTexture(textureUnit));
  glAssert(gl->glBindTexture(GL_TEXTURE_2D, fbo->texture()));
}

void FrameBufferObject::bindDepthTexture(unsigned int textureUnit)
{
  glAssert(gl->glActiveTexture(textureUnit));
  glAssert(gl->glBindTexture(GL_TEXTURE_2D, depthTexture));
}

void FrameBufferObject::resizeTexture(int texture, int width, int height,
                                      unsigned int component,
                                      unsigned int format)
{
  glAssert(gl->glBindTexture(GL_TEXTURE_2D, texture));
  glAssert(
      gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
  glAssert(
      gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
  // glAssert(
  // gl->glTexParameteri(GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE, GL_LUMINANCE));
  glAssert(gl->glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0,
                            component, GL_UNSIGNED_BYTE, NULL));
  glAssert(gl->glBindTexture(GL_TEXTURE_2D, 0));
}

