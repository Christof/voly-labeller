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
  }

  glAssert(fbo->bind());
  glAssert(glViewport(0, 0, width, height));

  if (!resize)
    glAssert(gl->glGenTextures(1, &depthTexture));

  glAssert(gl->glBindTexture(GL_TEXTURE_2D, depthTexture));
  glAssert(
      gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
  glAssert(
      gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
  glAssert(
      gl->glTexParameteri(GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE, GL_LUMINANCE));
  glAssert(gl->glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, width,
                            height, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE,
                            NULL));

  glAssert(gl->glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                                      GL_TEXTURE_2D, depthTexture, 0));

  if (resize)
  {
    glAssert(gl->glBindTexture(GL_TEXTURE_2D, fbo->texture()));
    glAssert(
        gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
    glAssert(
        gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
    glAssert(gl->glTexParameteri(GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE,
                                 GL_LUMINANCE));
    glAssert(gl->glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, width,
                              height, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE,
                              NULL));

    glAssert(gl->glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                        GL_TEXTURE_2D, fbo->texture(), 0));
  }

  fbo->release();
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

