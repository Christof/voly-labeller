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
  this->gl = gl;

  glAssert(gl->glGenFramebuffers(1, &framebuffer));
  glAssert(gl->glBindFramebuffer(GL_FRAMEBUFFER, framebuffer));

  glAssert(gl->glGenTextures(1, &depthTexture));
  resizeAndSetDepthAttachment(width, height);

  glAssert(gl->glGenTextures(1, &renderTexture));
  resizeAndSetColorAttachment(width, height);

  glAssert(gl->glBindTexture(GL_TEXTURE_2D, 0));

  auto status = gl->glCheckFramebufferStatus(GL_FRAMEBUFFER);
  if (status != GL_FRAMEBUFFER_COMPLETE)
    throw std::runtime_error("Framebuffer not complete " +
                             std::to_string(status));
}

void FrameBufferObject::resize(int width, int height)
{
  bind();

  resizeAndSetColorAttachment(width, height);
  resizeAndSetDepthAttachment(width, height);

  unbind();
}

void FrameBufferObject::bind()
{
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

void FrameBufferObject::resizeAndSetColorAttachment(int width, int height)
{
  resizeTexture(renderTexture, width, height, GL_RGBA, GL_RGBA8,
                GL_UNSIGNED_BYTE);
  glAssert(gl->glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                      GL_TEXTURE_2D, renderTexture, 0));
}

void FrameBufferObject::resizeAndSetDepthAttachment(int width, int height)
{
  resizeTexture(depthTexture, width, height, GL_DEPTH_COMPONENT,
                GL_DEPTH_COMPONENT32F, GL_FLOAT);
  glAssert(gl->glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                                      GL_TEXTURE_2D, depthTexture, 0));
}

void FrameBufferObject::resizeTexture(int texture, int width, int height,
                                      unsigned int component,
                                      unsigned int format, unsigned int type)
{
  glAssert(gl->glBindTexture(GL_TEXTURE_2D, texture));
  glAssert(
      gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
  glAssert(
      gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
  glAssert(gl->glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0,
                            component, type, NULL));
}

