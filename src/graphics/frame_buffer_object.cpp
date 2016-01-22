#include "./frame_buffer_object.h"
#include <QDebug>
#include <vector>
#include "./gl.h"

namespace Graphics
{

FrameBufferObject::FrameBufferObject()
  : colorTextures(layerCount), depthTextures(layerCount)
{
}

FrameBufferObject::~FrameBufferObject()
{
  glAssert(gl->glDeleteBuffers(1, &framebuffer));
  glAssert(gl->glDeleteBuffers(1, &depthTexture));
  glAssert(gl->glDeleteTextures(layerCount, colorTextures.data()));
  glAssert(gl->glDeleteTextures(layerCount, depthTextures.data()));
}

void FrameBufferObject::initialize(Gl *gl, int width, int height)
{
  this->gl = gl;

  glAssert(gl->glGenFramebuffers(1, &framebuffer));
  glAssert(gl->glBindFramebuffer(GL_FRAMEBUFFER, framebuffer));

  glAssert(gl->glGenTextures(1, &depthTexture));
  resizeAndSetDepthAttachment(width, height);

  glAssert(gl->glGenTextures(layerCount, colorTextures.data()));
  for (int i = 0; i < layerCount; ++i)
    resizeAndSetColorAttachment(colorTextures[i], GL_COLOR_ATTACHMENT0 + 2 * i,
                                width, height);

  glAssert(gl->glGenTextures(layerCount, depthTextures.data()));
  for (int i = 0; i < layerCount; ++i)
    resizeAndSetPositionAttachment(depthTextures[i],
                                   GL_COLOR_ATTACHMENT1 + 2 * i, width, height);

  glAssert(gl->glBindTexture(GL_TEXTURE_2D, 0));

  std::vector<GLenum> drawBuffers(2 * layerCount);
  for (int i = 0; i < 2 * layerCount; ++i)
    drawBuffers[i] = GL_COLOR_ATTACHMENT0 + i;
  glAssert(gl->glDrawBuffers(2 * layerCount, drawBuffers.data()));

  auto status = gl->glCheckFramebufferStatus(GL_FRAMEBUFFER);
  if (status != GL_FRAMEBUFFER_COMPLETE)
    throw std::runtime_error("Framebuffer not complete " +
                             std::to_string(status));

  unbind();
}

void FrameBufferObject::resize(int width, int height)
{
  bind();

  for (int i = 0; i < layerCount; ++i)
  {
    resizeAndSetColorAttachment(colorTextures[i], GL_COLOR_ATTACHMENT0 + 2 * i,
                                width, height);
    resizeAndSetPositionAttachment(depthTextures[i],
                                   GL_COLOR_ATTACHMENT1 + 2 * i, width, height);
  }

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

void FrameBufferObject::bindColorTexture(int index, unsigned int textureUnit)
{
  glAssert(gl->glActiveTexture(textureUnit));
  glAssert(gl->glBindTexture(GL_TEXTURE_2D, colorTextures[index]));
}

void FrameBufferObject::bindDepthTexture(int index, unsigned int textureUnit)
{
  glAssert(gl->glActiveTexture(textureUnit));
  glAssert(gl->glBindTexture(GL_TEXTURE_2D, depthTextures[index]));
}

void FrameBufferObject::bindDepthTexture(unsigned int textureUnit)
{
  glAssert(gl->glActiveTexture(textureUnit));
  glAssert(gl->glBindTexture(GL_TEXTURE_2D, depthTexture));
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

unsigned int FrameBufferObject::getColorTextureId(int index)
{
  return colorTextures[index];
}

unsigned int FrameBufferObject::getDepthTextureId(int index)
{
  return depthTextures[index];
}

unsigned int FrameBufferObject::getDepthTextureId()
{
  return depthTexture;
}

int FrameBufferObject::getLayerCount()
{
  return layerCount;
}

}  // namespace Graphics

