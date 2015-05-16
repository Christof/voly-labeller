#include "./frame_buffer_object.h"
#include <QDebug>
#include "./gl.h"

FrameBufferObject::~FrameBufferObject()
{
  glAssert(gl->glDeleteBuffers(1, &framebuffer));
  glAssert(gl->glDeleteTextures(1, &depthTexture));
  glAssert(gl->glDeleteTextures(1, &renderTexture));
}

void FrameBufferObject::initialize(Gl *gl, int width, int height)
{
  bool resize = framebuffer != 0;
  /*
  if (resize)
  {
    unbind();
    glAssert(gl->glDeleteBuffers(1, &framebuffer));
    glAssert(gl->glDeleteTextures(1, &depthTexture));
    glAssert(gl->glDeleteTextures(1, &renderTexture));
  }
  */
  if (!resize)
  {
    this->gl = gl;
    glAssert(gl->glGenFramebuffers(1, &framebuffer));
    glAssert(gl->glBindFramebuffer(GL_FRAMEBUFFER, framebuffer));

    glAssert(gl->glGenTextures(1, &depthTexture));
    resizeTexture(depthTexture, width, height, GL_DEPTH_COMPONENT,
                  GL_DEPTH_COMPONENT32F);

    glAssert(gl->glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER,
                                        GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D,
                                        depthTexture, 0));

    glAssert(gl->glGenTextures(1, &renderTexture));
    resizeTexture(renderTexture, width, height, GL_RGBA, GL_RGBA8);

    glAssert(gl->glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER,
                                        GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                                        renderTexture, 0));
    glAssert(gl->glBindTexture(GL_TEXTURE_2D, 0));

    auto status = gl->glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE)
    {
      switch (status)
      {
      case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
        throw std::runtime_error(
            "Framebuffer not complete: Incomplete attachment");
      case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
        throw std::runtime_error(
            "Framebuffer not complete: missing attachment");
      case GL_FRAMEBUFFER_UNSUPPORTED:
        throw std::runtime_error("Framebuffer not complete: unsupported");
      case GL_FRAMEBUFFER_UNDEFINED:
        throw std::runtime_error("Framebuffer not complete: undefined");
      case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
        throw std::runtime_error("Framebuffer not complete: draw buffer");
      case GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS:
        throw std::runtime_error("Framebuffer not complete: layer targets");
      }
      throw std::runtime_error("Framebuffer not complete " +
                               std::to_string(status));
    }
    // Set the list of draw buffers.
    qWarning() << "create fbo";
    /*
    fbo->release();
    fbo.release();
    */
  }
  else
  {
    bind();

    resizeTexture(depthTexture, width, height, GL_DEPTH_COMPONENT,
                  GL_DEPTH_COMPONENT32F);
    glAssert(gl->glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER,
                                        GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D,
                                        depthTexture, 0));


    resizeTexture(renderTexture, width, height, GL_RGBA, GL_RGBA8);
    glAssert(gl->glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER,
                                        GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                                        renderTexture, 0));

    int param = 0;
    gl->glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &param);
    qWarning() << "width of render texture" << param;

    unbind();
  }

  /*
  bind();

  qWarning() << "Resize to " << width << "x" << height;
  glAssert(gl->glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                                      GL_TEXTURE_2D, 0, 0));
  glAssert(gl->glBindTexture(GL_TEXTURE_2D, depthTexture));
  resizeTexture(depthTexture, width, height, GL_DEPTH_COMPONENT,
                GL_DEPTH_COMPONENT32F);
  glAssert(gl->glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                                      GL_TEXTURE_2D, depthTexture, 0));

  glAssert(gl->glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                      GL_TEXTURE_2D, 0, 0));
  glAssert(gl->glBindTexture(GL_TEXTURE_2D, renderTexture));
  resizeTexture(renderTexture, width, height, GL_RGBA, GL_RGBA8);
  glAssert(gl->glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                      GL_TEXTURE_2D, renderTexture, 0));

  unbind();
  */
}

void FrameBufferObject::bind()
{
  // QOpenGLFramebufferObject::blitFramebuffer(fbo.get(), fbo.get());
  glAssert(gl->glBindFramebuffer(GL_FRAMEBUFFER, framebuffer));
}

void FrameBufferObject::unbind()
{
  glAssert(gl->glBindFramebuffer(GL_FRAMEBUFFER, 0));
}

void FrameBufferObject::bindColorTexture(unsigned int textureUnit)
{
  glAssert(gl->glActiveTexture(textureUnit));
  glAssert(gl->glBindTexture(GL_TEXTURE_2D, renderTexture));
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
}

