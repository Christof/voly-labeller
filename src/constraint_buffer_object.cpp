#include "./constraint_buffer_object.h"

ConstraintBufferObject::~ConstraintBufferObject()
{
  glAssert(gl->glDeleteBuffers(1, &framebuffer));
}

void ConstraintBufferObject::initialize(Graphics::Gl *gl, int width, int height)
{
  this->gl = gl;
  this->width = width;
  this->height = height;

  glAssert(gl->glGenFramebuffers(1, &framebuffer));
  glAssert(gl->glBindFramebuffer(GL_FRAMEBUFFER, framebuffer));

  glAssert(gl->glGenTextures(1, &renderTexture));
  resizeAndSetColorAttachment(width, height);

  glAssert(gl->glBindTexture(GL_TEXTURE_2D, 0));

  GLenum drawBuffers[1] = { GL_COLOR_ATTACHMENT0 };
  glAssert(gl->glDrawBuffers(1, drawBuffers));

  auto status = gl->glCheckFramebufferStatus(GL_FRAMEBUFFER);
  if (status != GL_FRAMEBUFFER_COMPLETE)
    throw std::runtime_error("Framebuffer not complete " +
                             std::to_string(status));

  unbind();
}

void ConstraintBufferObject::bind()
{
  glAssert(gl->glBindFramebuffer(GL_FRAMEBUFFER, framebuffer));
}

void ConstraintBufferObject::unbind()
{
  glAssert(gl->glBindFramebuffer(GL_FRAMEBUFFER, 0));
}

void ConstraintBufferObject::bindTexture(unsigned int textureUnit)
{
  glAssert(gl->glActiveTexture(textureUnit));
  glAssert(gl->glBindTexture(GL_TEXTURE_2D, renderTexture));
}

unsigned int ConstraintBufferObject::getRenderTextureId()
{
  return renderTexture;
}

int ConstraintBufferObject::getWidth()
{
  return width;
}

int ConstraintBufferObject::getHeight()
{
  return height;
}

void ConstraintBufferObject::resizeAndSetColorAttachment(int width, int height)
{
  resizeTexture(renderTexture, width, height, GL_RED, GL_R8,
                GL_UNSIGNED_BYTE);
  glAssert(gl->glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                      GL_TEXTURE_2D, renderTexture, 0));
}

void ConstraintBufferObject::resizeTexture(int texture, int width, int height,
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

