#include "./frame_buffer_object.h"
#include <QDebug>
#include "./gl.h"

namespace Graphics
{

FrameBufferObject::~FrameBufferObject()
{
  glAssert(gl->glDeleteBuffers(1, &framebuffer));
  glAssert(gl->glDeleteTextures(1, &depthTexture));
  glAssert(gl->glDeleteTextures(1, &renderTexture));
  glAssert(gl->glDeleteTextures(1, &positionTexture));
  glAssert(gl->glDeleteTextures(1, &renderTexture2));
  glAssert(gl->glDeleteTextures(1, &positionTexture2));
}

void FrameBufferObject::initialize(Gl *gl, int width, int height)
{
  this->gl = gl;

  glAssert(gl->glGenFramebuffers(1, &framebuffer));
  glAssert(gl->glBindFramebuffer(GL_FRAMEBUFFER, framebuffer));

  glAssert(gl->glGenTextures(1, &depthTexture));
  resizeAndSetDepthAttachment(width, height);

  glAssert(gl->glGenTextures(1, &renderTexture));
  resizeAndSetColorAttachment(renderTexture, GL_COLOR_ATTACHMENT0, width,
                              height);

  glAssert(gl->glGenTextures(1, &positionTexture));
  resizeAndSetPositionAttachment(positionTexture, GL_COLOR_ATTACHMENT1, width,
                                 height);

  glAssert(gl->glGenTextures(1, &renderTexture2));
  resizeAndSetColorAttachment(renderTexture2, GL_COLOR_ATTACHMENT2, width,
                              height);

  glAssert(gl->glGenTextures(1, &positionTexture2));
  resizeAndSetPositionAttachment(positionTexture2, GL_COLOR_ATTACHMENT3, width,
                                 height);

  glAssert(gl->glBindTexture(GL_TEXTURE_2D, 0));

  GLenum drawBuffers[4] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1,
                            GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3 };
  glAssert(gl->glDrawBuffers(4, drawBuffers));

  auto status = gl->glCheckFramebufferStatus(GL_FRAMEBUFFER);
  if (status != GL_FRAMEBUFFER_COMPLETE)
    throw std::runtime_error("Framebuffer not complete " +
                             std::to_string(status));

  unbind();
}

void FrameBufferObject::resize(int width, int height)
{
  bind();

  resizeAndSetColorAttachment(renderTexture, GL_COLOR_ATTACHMENT0, width,
                              height);
  resizeAndSetPositionAttachment(positionTexture, GL_COLOR_ATTACHMENT1, width,
                                 height);
  resizeAndSetColorAttachment(renderTexture2, GL_COLOR_ATTACHMENT2, width,
                              height);
  resizeAndSetPositionAttachment(positionTexture2, GL_COLOR_ATTACHMENT3, width,
                                 height);
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

void FrameBufferObject::bindPositionTexture(unsigned int textureUnit)
{
  glAssert(gl->glActiveTexture(textureUnit));
  glAssert(gl->glBindTexture(GL_TEXTURE_2D, positionTexture));
}

void FrameBufferObject::bindDepthTexture(unsigned int textureUnit)
{
  glAssert(gl->glActiveTexture(textureUnit));
  glAssert(gl->glBindTexture(GL_TEXTURE_2D, depthTexture));
}

void FrameBufferObject::bindColorTexture2(unsigned int textureUnit)
{
  glAssert(gl->glActiveTexture(textureUnit));
  glAssert(gl->glBindTexture(GL_TEXTURE_2D, renderTexture2));
}

void FrameBufferObject::bindPositionTexture2(unsigned int textureUnit)
{
  glAssert(gl->glActiveTexture(textureUnit));
  glAssert(gl->glBindTexture(GL_TEXTURE_2D, positionTexture2));
}

void FrameBufferObject::resizeAndSetColorAttachment(int texture, int attachment,
                                                    int width, int height)
{
  resizeTexture(texture, width, height, GL_RGBA, GL_RGBA8, GL_UNSIGNED_BYTE);
  glAssert(gl->glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, attachment,
                                      GL_TEXTURE_2D, texture, 0));
}

void FrameBufferObject::resizeAndSetPositionAttachment(int texture,
                                                       int attachment,
                                                       int width, int height)
{
  resizeTexture(texture, width, height, GL_RGBA, GL_RGBA32F, GL_FLOAT);
  glAssert(gl->glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, attachment,
                                      GL_TEXTURE_2D, texture, 0));
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

unsigned int FrameBufferObject::getRenderTextureId()
{
  return renderTexture;
}

unsigned int FrameBufferObject::getPositionTextureId()
{
  return positionTexture;
}

unsigned int FrameBufferObject::getDepthTextureId()
{
  return depthTexture;
}

}  // namespace Graphics
