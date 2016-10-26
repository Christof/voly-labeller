#include "./frame_buffer_object.h"
#include <QDebug>
#include <vector>
#include "./gl.h"

namespace Graphics
{

FrameBufferObject::FrameBufferObject(unsigned int layerCount)
  : layerCount(layerCount)
{
}

FrameBufferObject::~FrameBufferObject()
{
  glAssert(gl->glDeleteBuffers(1, &framebuffer));
  glAssert(gl->glDeleteBuffers(1, &depthTexture));
  glAssert(gl->glDeleteTextures(1, &colorTexturesArray));
  glAssert(gl->glDeleteTextures(1, &accumulatedLayersTexture));
}

void FrameBufferObject::initialize(Gl *gl, int width, int height)
{
  this->gl = gl;

  glAssert(gl->glGenFramebuffers(1, &framebuffer));
  glAssert(gl->glBindFramebuffer(GL_FRAMEBUFFER, framebuffer));

  glAssert(gl->glGenTextures(1, &depthTexture));
  resizeAndSetDepthAttachment(width, height);

  glAssert(gl->glGenTextures(1, &colorTexturesArray));
  resizeAndSetColorArrayAttachment(colorTexturesArray, GL_COLOR_ATTACHMENT0,
                                   width, height);

  glAssert(gl->glGenTextures(1, &accumulatedLayersTexture));
  resizeAndSetColorAttachment(accumulatedLayersTexture,
                              GL_COLOR_ATTACHMENT0 + layerCount, width, height);

  glAssert(gl->glBindTexture(GL_TEXTURE_2D, 0));

  std::vector<GLenum> drawBuffers(layerCount + 1);
  for (unsigned int i = 0; i < layerCount + 1; ++i)
    drawBuffers[i] = GL_COLOR_ATTACHMENT0 + i;
  glAssert(gl->glDrawBuffers(layerCount + 1, drawBuffers.data()));

  auto status = gl->glCheckFramebufferStatus(GL_FRAMEBUFFER);
  if (status != GL_FRAMEBUFFER_COMPLETE)
    throw std::runtime_error("Framebuffer not complete " +
                             std::to_string(status));

  unbind();
}

void FrameBufferObject::resize(int width, int height)
{
  bind();

  resizeAndSetDepthAttachment(width, height);

  // TODO(all): investigate why deleting and generating a new texture is
  // necessary. If this is not done, an GL_INVALID_OPERATION causes the
  // application to crash.
  // Before switching to a 3D texture for the colors, this was not necessary.
  gl->glDeleteTextures(1, &accumulatedLayersTexture);
  gl->glGenTextures(1, &accumulatedLayersTexture);
  resizeAndSetColorAttachment(accumulatedLayersTexture,
                              GL_COLOR_ATTACHMENT0 + layerCount, width, height);
  resizeAndSetColorArrayAttachment(colorTexturesArray, GL_COLOR_ATTACHMENT0,
                                   width, height);

  glAssert(gl->glBindTexture(GL_TEXTURE_2D, 0));

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
  glAssert(gl->glBindTexture(GL_TEXTURE_3D, colorTexturesArray));
}

void FrameBufferObject::bindAccumulatedLayersTexture(unsigned int textureUnit)
{
  glAssert(gl->glActiveTexture(textureUnit));
  glAssert(gl->glBindTexture(GL_TEXTURE_2D, accumulatedLayersTexture));
}

void FrameBufferObject::bindDepthTexture(unsigned int textureUnit)
{
  glAssert(gl->glActiveTexture(textureUnit));
  glAssert(gl->glBindTexture(GL_TEXTURE_2D, depthTexture));
}

void FrameBufferObject::resizeAndSetColorAttachment(unsigned int texture,
                                                    int attachment, int width,
                                                    int height)
{
  resizeTexture(texture, width, height, GL_RGBA, GL_RGBA16F, GL_FLOAT);
  glAssert(gl->glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, attachment,
                                      GL_TEXTURE_2D, texture, 0));
}

void FrameBufferObject::resizeAndSetColorArrayAttachment(
    unsigned int texture, unsigned int attachment, int width, int height)
{
  gl->glBindTexture(GL_TEXTURE_3D, texture);
  gl->glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  gl->glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  gl->glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA16F, width, height, layerCount, 0,
                   GL_RGBA, GL_FLOAT, nullptr);

  for (unsigned int layerIndex = 0; layerIndex < layerCount; ++layerIndex)
    glAssert(gl->glFramebufferTexture3D(GL_DRAW_FRAMEBUFFER,
                                        attachment + layerIndex, GL_TEXTURE_3D,
                                        texture, 0, layerIndex));

  gl->glBindTexture(GL_TEXTURE_3D, 0);
}

void FrameBufferObject::resizeAndSetDepthAttachment(int width, int height)
{
  resizeTexture(depthTexture, width, height, GL_DEPTH_STENCIL,
                GL_DEPTH24_STENCIL8, GL_UNSIGNED_INT_24_8);
  glAssert(gl->glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER,
                                      GL_DEPTH_STENCIL_ATTACHMENT,
                                      GL_TEXTURE_2D, depthTexture, 0));
}

void FrameBufferObject::resizeTexture(unsigned int texture, int width,
                                      int height, unsigned int component,
                                      unsigned int format, unsigned int type)
{
  gl->glBindTexture(GL_TEXTURE_2D, texture);
  gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  gl->glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, component, type,
                   nullptr);
}

unsigned int FrameBufferObject::getColorTextureId()
{
  return colorTexturesArray;
}

unsigned int FrameBufferObject::getAccumulatedLayersTextureId()
{
  return accumulatedLayersTexture;
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

